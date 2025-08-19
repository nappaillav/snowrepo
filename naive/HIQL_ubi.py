import copy
import random
import time

import numpy as np
import torch
from gametrackr.core import get_arguments, get_class, instantiate_class
from gametrackr.core.pytorch.dataset_complex_relabel import (
    PytorchEpisodeFrameDatasetComplexRelabel,
)
from gametrackr.core.pytorch.tools import omegaconf_to_conf
from torch.utils.data import DataLoader
from tqdm import tqdm


def _tqdm(iter, desc="", verbose=True):
    if verbose:
        return tqdm(iter, desc=desc)
    else:
        return iter


# useful to seed parallel dataloader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_empty_log_dicts():
    losses = {"v1": [], "v2": [], "policy": [], "high_policy": [], "low_policy": []}
    value_infos = {
        "accept_prob": [],
        "v1_min": [],
        "v1_max": [],
        "v1_mean": [],
        "abs adv mean": [],
        "adv mean": [],
        "adv max": [],
        "adv min": [],
    }
    policy_infos = {
        "bc_log_probs": [],
        "adv": [],
        "mse": [],
        "high_bc_log_probs": [],
        "high_adv": [],
        "high_scale": [],
        "high_mse": [],
    }
    return losses, value_infos, policy_infos


class HIQLValueFrameRelabeler:
    def __init__(
        self,
        mapping,
        p_current,
        p_future,
        p_random,
        update_reward=False,
        geom_sample=False,
    ):
        self._mapping = mapping
        self.goal_types = ["current", "future", "random"]
        self.probs = [p_current, p_future, p_random]
        self.update_reward = update_reward
        self.geom_sample = geom_sample

    def __call__(self, episode, ep_id, t, frame_size, dataset):
        goal_type = np.random.choice(self.goal_types, p=self.probs)
        if goal_type == "current":
            idx = t
            source_episode = episode
        elif goal_type == "future":
            ep_len = episode[next(episode.keys().__iter__())].size()[0]
            if self.geom_sample:
                idx = min(
                    t
                    + np.ceil(np.log(1 - np.random.rand()) / np.log(0.99)).astype(int),
                    ep_len - 1,
                )
            else:
                idx = np.random.randint(
                    min(t + 1, ep_len - 1), ep_len
                )  # idx in range [t+1, T]
            source_episode = episode
        else:  # goal_type == "random"
            rnd_frame_idx = np.random.randint(len(dataset))
            _id, idx = dataset._index[rnd_frame_idx]
            source_episode = dataset._get_episode(_id)

        if self.update_reward:
            if goal_type == "current" or goal_type == "future":
                if idx == t:
                    episode["reward"][t] = 1.0
                else:
                    episode["reward"][t] = 0.0
            else:  # random episode, so most likely not same frame
                if idx == t and _id[1] == ep_id[1]:
                    episode["reward"][t] = 1.0
                else:
                    episode["reward"][t] = 0.0

        for k in self._mapping:
            _from = k["from"]
            _to = k["to"]
            episode[_to] = source_episode[_from][idx].unsqueeze(0)
        return episode


class HighPolicyFrameRelabeler:
    def __init__(self, mapping, p_current, p_future, p_random, way_steps):
        self._mapping = mapping
        self.goal_types = ["current", "future", "random"]
        self.probs = [p_current, p_future, p_random]
        self.way_steps = way_steps

    def __call__(self, episode, ep_id, t, frame_size, dataset):
        goal_type = np.random.choice(self.goal_types, p=self.probs)
        ep_len = episode[next(episode.keys().__iter__())].size()[0]
        if goal_type == "current":
            raise NotImplementedError
        elif goal_type == "future":
            goal_idx = np.random.randint(
                min(t + 1, ep_len - 1), ep_len
            )  # idx in range [t+1, T]
            goal_episode = episode
            # compute high_target, i.e. the target "action" (waypoint) associated with sampled high level goal
            high_target_idx = min(t + self.way_steps, goal_idx)
        else:  # goal_type == "random"
            rnd_frame_idx = np.random.randint(len(dataset))
            _id, goal_idx = dataset._index[rnd_frame_idx]
            goal_episode = dataset._get_episode(_id)
            high_target_idx = min(t + self.way_steps, ep_len - 1)

        for k in self._mapping:
            _from = k["from"]
            _to = k["to"]
            episode[_to] = goal_episode[_from][goal_idx].unsqueeze(0)
        episode["high_target"] = episode[_from][high_target_idx].unsqueeze(0)
        return episode


# Used to compute goals for the low level policy (waypoints)
class WaypointFrameRelabeler:
    def __init__(self, way_steps):
        self.way_steps = way_steps

    def __call__(self, episode, ep_id, t, frame_size, dataset):
        ep_len = episode[next(episode.keys().__iter__())].size()[0]
        idx = min(t + self.way_steps, ep_len - 1)
        episode["low_goal"] = episode["obs"][idx].unsqueeze(0)
        return episode


def expectile_loss(adv, diff, expectile):
    weight = torch.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


# This is HIQL w/o representations (not vanilla HIQL)
# WORK IN PROGRESS - PERFORMANCES ARE STILL BELOW ORIGINAL IMPLEMENTATION - IDEAS OF IMPROVEMENTS WELCOMED
def run_hiql(training_db, model_db, cfg, logger=None, verbose=True):
    cfg = omegaconf_to_conf(
        cfg
    )  # Avoid costly Hydra config lookups by dumping config data into namespace
    if verbose:
        print("Launching HIQL (w/o repr) training")

    while len(training_db) < 1:
        print("Wait for episodes...")
        time.sleep(1)

    # SETTING DATASET
    frame_size = 1
    dataset = PytorchEpisodeFrameDatasetComplexRelabel(
        training_db,
        frame_size=frame_size,
        padding_begin=frame_size - 1,
        padding_end=0,
        episode_relabellers=[
            HIQLValueFrameRelabeler(
                cfg.mapping_value, 0.2, 0.5, 0.3, update_reward=True, geom_sample=True
            ),
            HighPolicyFrameRelabeler(
                cfg.mapping_high_policy,
                0.0,
                1 - cfg.high_p_randomgoal,
                cfg.high_p_randomgoal,
                cfg.way_steps,
            ),
            WaypointFrameRelabeler(cfg.way_steps),
        ],
    )

    # SETTING MODELS
    policy = instantiate_class(cfg.policy)
    vf1 = instantiate_class(cfg.vf)
    vf2 = instantiate_class(cfg.vf)

    policy.to(cfg.device)
    vf1.to(cfg.device)
    vf2.to(cfg.device)

    print(policy)
    print(vf1)

    target_vf1 = copy.deepcopy(vf1)
    target_vf2 = copy.deepcopy(vf2)

    # SETTING OPTIMIZERS
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    policy_optimizer = get_class(cfg.algorithm.optimizer)(
        policy.parameters(), **optimizer_args
    )
    vf1_optimizer = get_class(cfg.algorithm.optimizer)(
        vf1.parameters(), **optimizer_args
    )
    vf2_optimizer = get_class(cfg.algorithm.optimizer)(
        vf2.parameters(), **optimizer_args
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    gradient_step = 0

    # save first model (it's probably cleaner than creating a RandomBot in the offline_trainer, to be discussed)
    if verbose:
        print("Saving model...")
    m = copy.deepcopy(policy)
    m.to("cpu")
    model_db.push("model", m)

    losses, value_infos, policy_infos = get_empty_log_dicts()
    for epoch in range(cfg.max_epoch):
        if epoch == 0 or (
            not cfg.reindex_every is None and epoch % cfg.reindex_every == 0
        ):
            dataset._run_index()
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.algorithm.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                persistent_workers=cfg.num_workers > 0,
                pin_memory=cfg.pin_memory,
                worker_init_fn=seed_worker,
                generator=g,
            )

        _start_t = time.time()
        dataloader_st = _start_t
        dataloader_time = 0
        n = 0
        for frame_batch in _tqdm(
            dataloader, desc="Epoch " + str(epoch), verbose=verbose
        ):
            dataloader_time += time.time() - dataloader_st

            frame_batch = {k: v.to(cfg.device) for k, v in frame_batch.items()}
            batch = {
                "next_obs": frame_batch["next_obs"].squeeze(),
                "rewards": frame_batch["reward"],
                "actions": frame_batch["action"].squeeze(),
                "goals": frame_batch["goal"].squeeze(),
                "low_goals": frame_batch["low_goal"].squeeze()
                if cfg.use_waypoints
                else frame_batch["high_goal"].squeeze(),
                "high_goals": frame_batch["high_goal"].squeeze(),
                "high_targets": frame_batch["high_target"].squeeze(),
            }
            if cfg.split_obs:
                batch["obs"] = torch.cat(
                    (frame_batch["obs/pos"], frame_batch["obs/other"]), dim=2
                ).squeeze()
            else:
                batch["obs"] = frame_batch["obs"].squeeze()

            # vf_loss vanilla GC-IQL (vanilla == using action free vf as in HIQL, rather than qf & vf as in IQL paper)
            masks = 1.0 - batch["rewards"]  # masks are 0 if terminal, 1 otherwise
            rewards = batch["rewards"] - 1.0
            next_v1_t, next_v2_t = [
                v(batch["next_obs"], batch["goals"]) for v in [target_vf1, target_vf2]
            ]
            next_v_t = torch.min(next_v1_t, next_v2_t).detach()
            q_t = (
                cfg.algorithm.reward_scale * rewards
                + masks * cfg.algorithm.discount * next_v_t
            )

            v_t = (
                target_vf1(batch["obs"], batch["goals"])
                + target_vf2(batch["obs"], batch["goals"])
            ) / 2
            adv = (q_t - v_t).detach()

            q1_t = (
                cfg.algorithm.reward_scale * rewards
                + masks * cfg.algorithm.discount * next_v1_t
            )
            q2_t = (
                cfg.algorithm.reward_scale * rewards
                + masks * cfg.algorithm.discount * next_v2_t
            )
            v1, v2 = vf1(batch["obs"], batch["goals"]), vf2(
                batch["obs"], batch["goals"]
            )

            vf1_loss = expectile_loss(
                adv, q1_t.detach() - v1, cfg.algorithm.expectile
            ).mean()
            vf2_loss = expectile_loss(
                adv, q2_t.detach() - v2, cfg.algorithm.expectile
            ).mean()

            adv = adv.cpu().detach()
            v1 = v1.cpu().detach()
            value_infos["abs adv mean"].append(torch.abs(adv).mean())
            value_infos["adv mean"].append(adv.mean())
            value_infos["adv max"].append(adv.max())
            value_infos["adv min"].append(adv.min())
            value_infos["accept_prob"].append((adv >= 0).float().mean())
            value_infos["v1_min"].append(v1.min())
            value_infos["v1_max"].append(v1.max())
            value_infos["v1_mean"].append(v1.mean())

            # low-level / flat policy_loss
            vf_pred = (
                vf1(batch["obs"], batch["low_goals"])
                + vf2(batch["obs"], batch["low_goals"])
            ) / 2
            next_vf1 = vf1(batch["next_obs"], batch["low_goals"])
            next_vf2 = vf2(batch["next_obs"], batch["low_goals"])
            next_vf_pred = (next_vf1 + next_vf2) / 2
            adv = (next_vf_pred - vf_pred).detach()
            exp_adv = torch.exp(adv * cfg.algorithm.beta)
            exp_adv = torch.clamp(exp_adv, max=cfg.algorithm.clip_score)
            weights = exp_adv[:, 0].detach()
            if cfg.use_waypoints:
                mean, dist = policy.low.forward(
                    torch.cat((batch["low_goals"], batch["obs"]), dim=1)
                )
                logpp = dist.log_prob(batch["actions"])
                low_policy_loss = (-logpp * weights).mean()
            else:
                mean, dist = policy.forward(
                    torch.cat((batch["low_goals"], batch["obs"]), dim=1)
                )
                logpp = dist.log_prob(batch["actions"])
                policy_loss = (-logpp * weights).mean()

            policy_weights = weights
            policy_infos["bc_log_probs"].append(logpp.cpu().detach().mean())
            policy_infos["adv"].append(adv.cpu().detach().mean())
            policy_infos["mse"].append(
                ((mean.detach() - batch["actions"].detach()) ** 2).cpu().mean()
            )

            # high-level policy_loss
            if cfg.use_waypoints:
                vf_pred = (
                    vf1(batch["obs"], batch["high_goals"])
                    + vf2(batch["obs"], batch["high_goals"])
                ) / 2
                next_vf1 = vf1(batch["high_targets"], batch["high_goals"])
                next_vf2 = vf2(batch["high_targets"], batch["high_goals"])
                next_vf_pred = (next_vf1 + next_vf2) / 2
                adv = next_vf_pred - vf_pred
                exp_adv = torch.exp(adv * cfg.algorithm.high_beta)
                exp_adv = torch.clamp(exp_adv, max=cfg.algorithm.clip_score)
                weights = exp_adv[:, 0].detach()
                mean, dist = policy.high.forward(
                    torch.cat((batch["high_goals"], batch["obs"]), dim=1)
                )
                # in HIQL w/o repr learning they use "high_targets - obs" as target
                # maybe because relative positions are easier to learn than absolute ones (smaller values)
                # to "fix" this at inference time they add obs back to the high_pi output before feeding low pi
                target = batch["high_targets"] - batch["obs"]
                logpp = dist.log_prob(target)
                high_policy_loss = (-logpp * weights).mean()
                policy_loss = low_policy_loss + high_policy_loss

                policy_infos["high_bc_log_probs"].append(logpp.cpu().detach().mean())
                policy_infos["high_adv"].append(adv.cpu().detach().mean())
                policy_infos["high_scale"].append(
                    dist.base_dist.scale.diag().cpu().detach().mean()
                )
                policy_infos["high_mse"].append(
                    ((mean.detach() - target.detach()) ** 2).cpu().mean()
                )

            # Update networks
            if gradient_step % cfg.algorithm.v_update_period == 0:
                vf1_optimizer.zero_grad()
                vf1_loss.backward()
                vf1_optimizer.step()

                vf2_optimizer.zero_grad()
                vf2_loss.backward()
                vf2_optimizer.step()

            if gradient_step % cfg.algorithm.policy_update_period == 0:
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

            # Soft updates
            if gradient_step % cfg.algorithm.target_update_period == 0:
                for target_param, param in zip(
                    target_vf1.parameters(), vf1.parameters()
                ):
                    target_param.data.copy_(
                        cfg.algorithm.polyak_coef * param.data
                        + (1.0 - cfg.algorithm.polyak_coef) * target_param.data
                    )

                for target_param, param in zip(
                    target_vf2.parameters(), vf2.parameters()
                ):
                    target_param.data.copy_(
                        cfg.algorithm.polyak_coef * param.data
                        + (1.0 - cfg.algorithm.polyak_coef) * target_param.data
                    )

            losses["v1"].append(vf1_loss.item())
            losses["v2"].append(vf2_loss.item())
            losses["policy"].append(policy_loss.item())
            if cfg.use_waypoints:
                losses["high_policy"].append(high_policy_loss.item())
                losses["low_policy"].append(low_policy_loss.item())

            gradient_step += 1
            dataloader_st = time.time()

            # trying to log as close as possible as what is done in HIQL paper
            if gradient_step % cfg.log_every == 0:
                # log histogram of policy weights (i.e. scaled and exponentiated adv, and clipped)
                logger.add_histogram(name='histogram_awr_policy', values=policy_weights, epoch=epoch)
                for k, v in losses.items():
                    logger.add_scalar(
                        "loss/" + k, v[-1] if len(v) > 0 else np.NaN, epoch
                    )
                    logger.add_scalar("infos/dataset_size", len(dataset), epoch)
                for k, v in value_infos.items():
                    logger.add_scalar(
                        "value/" + k, v[-1] if len(v) > 0 else np.NaN, epoch
                    )
                for k, v in policy_infos.items():
                    logger.add_scalar(
                        "policy/" + k, v[-1] if len(v) > 0 else np.NaN, epoch
                    )
                losses, value_infos, policy_infos = get_empty_log_dicts()

        # for k, v in losses.items():
        #     logger.add_scalar("loss/" + k, np.mean(v), epoch)
        #     logger.add_scalar("infos/dataset_size", len(dataset), epoch)
        # for k, v in value_infos.items():
        #     logger.add_scalar("value/" + k, np.mean(v), epoch)
        # for k, v in policy_infos.items():
        #     logger.add_scalar("policy/" + k, np.mean(v), epoch)

        logger.add_scalar(
            "infos/batches_per_second", n / (time.time() + 0.001 - _start_t), epoch
        )
        logger.add_scalar("infos/time_for_each_epoch", time.time() - _start_t, epoch)
        logger.add_scalar(
            "infos/dataloader_time_for_each_epoch", dataloader_time, epoch
        )
        epoch += 1
        if epoch % cfg.save_every == 0:
            if verbose:
                print("Saving model...")
            m = copy.deepcopy(policy)
            m.to("cpu")
            model_db.push("model", m)
    m = copy.deepcopy(policy)
    m.to("cpu")
    return m
