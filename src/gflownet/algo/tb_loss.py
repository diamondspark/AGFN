from torch_scatter import scatter, scatter_sum
import torch

def compute_batch_losses(algo, gfn_trainer, batch, dev, hps, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0.
        t : int
          iteration number"""
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        # print('tbloss.py num_trajs', num_trajs)
        log_rewards = batch.log_rewards
        # print('TBloss.py log_rewawrds', log_rewards, log_rewards.shape)
        # Clip rewards
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(log_rewards, torch.tensor(algo.illegal_action_logreward, device=dev)).float()
        invalid_mask = 1 - batch.is_valid
        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the first graph of each trajectory
        first_graph_idx = torch.zeros_like(batch.traj_lens)
        torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical representing the forward
        # policy P_F, optionally a backward policy P_B, and per-graph outputs (e.g. F(s) in SubTB).
        if algo.p_b_is_parameterized:
            fwd_cat, bck_cat, per_graph_out, logZ_forward = gfn_trainer.model(batch)
        else:
            fwd_cat, per_graph_out, logZ_forward = gfn_trainer.model(batch)

        # Compute trajectory balance objective
        # log_Z = gfn_trainer.model.module.logZ(cond_info)[:, 0]
        # print('cond_info ',cond_info.shape, ' module logZ ',gfn_trainer.model.module.logZ(cond_info).shape)
        log_Z = logZ_forward[first_graph_idx]  # gfn_trainer.model.logZ_forward#(cond_info)[:, 0]
        # Compute the log prob of each action in the trajectory
        if algo.correct_idempotent:
            # If we want to correct for idempotent actions, we need to sum probabilities
            # i.e. to compute P(s' | s) = sum_{a that lead to s'} P(a|s)
            # here we compute the indices of the graph that each action corresponds to, ip_lens
            # contains the number of idempotent actions for each transition, so we
            # repeat_interleave as with batch_idx
            ip_batch_idces = torch.arange(batch.ip_lens.shape[0], device=dev).repeat_interleave(batch.ip_lens)
            # Indicate that the `batch` corresponding to each action is the above
            ip_log_prob = fwd_cat.log_prob(batch.ip_actions, batch=ip_batch_idces)
            # take the logsumexp (because we want to sum probabilities, not log probabilities)
            # TODO: numerically stable version:
            p = scatter(ip_log_prob.exp(), ip_batch_idces, dim=0, dim_size=batch_idx.shape[0], reduce="sum")
            # As a (reasonable) band-aid, ignore p < 1e-30, this will prevent underflows due to
            # scatter(small number) = 0 on CUDA
            log_p_F = p.clamp(1e-30).log()

            if algo.p_b_is_parameterized:
                # Now we repeat this but for the backward policy
                bck_ip_batch_idces = torch.arange(batch.bck_ip_lens.shape[0], device=dev).repeat_interleave(
                    batch.bck_ip_lens
                )
                bck_ip_log_prob = bck_cat.log_prob(batch.bck_ip_actions, batch=bck_ip_batch_idces)
                bck_p = scatter(
                    bck_ip_log_prob.exp(), bck_ip_batch_idces, dim=0, dim_size=batch_idx.shape[0], reduce="sum"
                )
                log_p_B = bck_p.clamp(1e-30).log()
        else:
            # Else just naively take the logprob of the actions we took
            log_p_F = fwd_cat.log_prob(batch.actions)
            if algo.p_b_is_parameterized:
                log_p_B = bck_cat.log_prob(batch.bck_actions)

        if algo.p_b_is_parameterized:
            # If we're modeling P_B then trajectories are padded with a virtual terminal state sF,
            # zero-out the logP_F of those states
            log_p_F[final_graph_idx] = 0
            if algo.is_doing_subTB:
                # Force the pad states' F(s) prediction to be R
                per_graph_out[final_graph_idx, 0] = clip_log_R

            # To get the correct P_B we need to shift all predictions by 1 state, and ignore the
            # first P_B prediction of every trajectory.
            # Our batch looks like this:
            # [(s1, a1), (s2, a2), ..., (st, at), (sF, None),   (s1, a1), ...]
            #                                                   ^ new trajectory begins
            # For the P_B of s1, we need the output of the model at s2.

            # We also have access to the is_sink attribute, which tells us when P_B must = 1, which
            # we'll use to ignore the last padding state(s) of each trajectory. This by the same
            # occasion masks out the first P_B of the "next" trajectory that we've shifted.
            log_p_B = torch.cat([log_p_B[1:], log_p_B[:1]]) * (1 - batch.is_sink)
        else:
            log_p_B = batch.log_p_B
            print('tbloss logpb ', log_p_B, log_p_B.shape)
            print('tbloss logpF ', log_p_F, log_p_F.shape)
        assert log_p_F.shape == log_p_B.shape

        # This is the log probability of each trajectory
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        
        if (gfn_trainer.hps['offline_data']) and (gfn_trainer.hps['MLE_coeff']>0):
            mle_loss = -traj_log_p_F[batch.num_sub_online_trajs:]#.mean()  #num_sub_online_trajs: = sub offline trajectories
        else:
            mle_loss = 0

        if algo.is_doing_subTB:
            # SubTB interprets the per_graph_out predictions to predict the state flow F(s)
            traj_losses = self.subtb_loss_fast(log_p_F, log_p_B, per_graph_out[:, 0], clip_log_R, batch.traj_lens)
            # The position of the first graph of each trajectory
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            first_graph_idx = torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            log_Z = per_graph_out[first_graph_idx, 0]
        else:
            # Compute log numerator and denominator of the TB objective
            # print(f'logZ {logZ.shape} traj_log_p_F {traj_log_p_F.shape}')
            numerator = log_Z + traj_log_p_F
            denominator = clip_log_R + traj_log_p_B

            if algo.mask_invalid_rewards:
                # Instead of being rude to the model and giving a
                # logreward of -100 what if we say, whatever you think the
                # logprobablity of this trajetcory is it should be smaller
                # (thus the `numerator - 1`). Why 1? Intuition?
                denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

            if algo.epsilon is not None:
                # Numerical stability epsilon
                epsilon = torch.tensor([algo.epsilon], device=dev).float()
                numerator = torch.logaddexp(numerator, epsilon)
                denominator = torch.logaddexp(denominator, epsilon)
            if algo.tb_loss_is_mae:
                traj_losses = abs(numerator - denominator)
            elif algo.tb_loss_is_huber:
                pass  # TODO
            else:
                traj_losses = (numerator - denominator).pow(2)

        # Normalize losses by trajectory length
        if algo.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        if algo.reward_normalize_losses:
            # multiply each loss by how important it is, using R as the importance factor
            # factor = Rp.exp() / Rp.exp().sum()
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if algo.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            if algo.reward_loss_is_mae:
                reward_losses = abs(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap])
            else:
                reward_losses = (log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap]).pow(2)
            reward_loss = reward_losses.mean()
        else:
            reward_loss = 0

        # mle_min, mle_max = 0, 500 #  min(mle_loss), max(mle_loss)
        # gfn_min, gfn_max = 0, 120000 #min(traj_losses), max(traj_losses)

        # gfn_scaled = (traj_losses - gfn_min)/(gfn_max - gfn_min)
        # mle_scaled = (mle_loss - mle_min)/(mle_max - mle_min)

        # loss = (gfn_trainer.hps['gfn_lambda'])*gfn_scaled.mean()  + (1-gfn_trainer.hps['gfn_lambda'])*mle_scaled.mean()
        if gfn_trainer.hps['MLE_coeff']>0:
            loss = gfn_trainer.hps['gfn_loss_coeff']*traj_losses.mean() + gfn_trainer.hps['MLE_coeff']*mle_loss.mean()
        else:
            loss = gfn_trainer.hps['gfn_loss_coeff']*traj_losses.mean() 

        # print('tbloss loss ', loss)

        # loss = gfn_trainer.hps['gfn_loss_coeff']*traj_losses.mean()*t**(0.33) + reward_loss * algo.reward_loss_multiplier + gfn_trainer.hps['MLE_coeff']*mle_loss*t**(-0.33)
        # loss = gfn_trainer.hps['gfn_loss_coeff']*traj_losses.mean() + reward_loss * algo.reward_loss_multiplier + gfn_trainer.hps['MLE_coeff']*mle_loss

        # print('tbloss.py fltgfn loss ',traj_losses.mean(), ' flatmle loss ',mle_loss)
        info = {
            "offline_loss": traj_losses[: batch.num_offline].mean().item() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean().item() if batch.num_online > 0 else 0,
            # "reward_loss": reward_loss,
            "invalid_trajectories": (invalid_mask.sum() / batch.num_online).item() if batch.num_online > 0 else 0,
            "invalid_logprob": ((invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4)).item(),
            "invalid_losses": ((invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4)).item(),
            "logZ": log_Z.mean().item(),
            "loss": loss.item(),
            "mle_loss":mle_loss.mean().item() if gfn_trainer.hps['MLE_coeff']>0 else 0,
            "total_gfn_loss":traj_losses.mean().item(),
            # "total_gfn_loss_scaled":gfn_scaled.mean().item(),
            # "mle_loss_scaled": mle_scaled.mean().item() if gfn_trainer.hps['MLE_coeff']>0 else 0,
        }
        # print('tb loss.py info ', info)

        return loss, info

