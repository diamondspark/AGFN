from torch_scatter import scatter, scatter_sum
import torch

def compute_batch_losses(algo, gfn_trainer, batch, dev, t, num_bootstrap: int = 0):
    num_trajs = int(batch.traj_lens.shape[0])
    # print('rtbloss.py compte_batch_loss batch ', batch.__dict__.keys())
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
    
    fwd_cat_post, logZ_forward_post = gfn_trainer.model(batch)
    # logZ_forward_post = logZ_forward_post[:,0]
    log_Z = logZ_forward_post[first_graph_idx]
    log_p_F = fwd_cat_post.log_prob(batch.actions)  


    # This is the log probability of each trajectory
    traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

    if (gfn_trainer.hps['offline_data']) and (gfn_trainer.hps['MLE_coeff']>0):
        mle_loss = -traj_log_p_F[batch.num_sub_online_trajs:]#.mean()  #num_sub_online_trajs: = sub offline trajectories
    else:
        mle_loss = 0

    with torch.no_grad():
        fwd_cat_prior = gfn_trainer.model_prior(batch)
    log_p_F_prior = fwd_cat_prior.log_prob(batch.actions)
    traj_log_p_F_prior = scatter(log_p_F_prior, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
    
    numerator = log_Z + traj_log_p_F
    denominator = traj_log_p_F_prior + clip_log_R

    if gfn_trainer.hps['vargrad']:
        numerator = log_Z_hat = -traj_log_p_F + traj_log_p_F_prior + clip_log_R
        denominator = log_Z_hat.mean()

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
        if gfn_trainer.hps['vargrad']:
            traj_losses = (numerator - denominator).pow(2)#.mean()
        else:
            traj_losses = (numerator - denominator).pow(2)
        # print('rtbloss.py trajlosses ', traj_losses, traj_losses.shape)

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

    if (gfn_trainer.hps['offline_data']) and (gfn_trainer.hps['MLE_coeff']>0):
        loss = gfn_trainer.hps['gfn_loss_coeff']*traj_losses.mean() + gfn_trainer.hps['MLE_coeff']*mle_loss.mean()
    else:
        loss = gfn_trainer.hps['gfn_loss_coeff']*traj_losses.mean() 

    info = {
            "offline_loss": traj_losses[: batch.num_offline].mean().item() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean().item() if batch.num_online > 0 else 0,
            # "reward_loss": reward_loss,
            "invalid_trajectories": (invalid_mask.sum() / batch.num_online).item() if batch.num_online > 0 else 0,
            "invalid_logprob": ((invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4)).item(),
            "invalid_losses": ((invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4)).item(),
            "logZ": log_Z.mean(),#.item(),
            "loss": loss.item(),
            "mle_loss":mle_loss.mean().item() if (gfn_trainer.hps['offline_data']) and (gfn_trainer.hps['MLE_coeff']>0) else 0,
            "total_gfn_loss":traj_losses.mean().item(),

        }

    return loss, info
