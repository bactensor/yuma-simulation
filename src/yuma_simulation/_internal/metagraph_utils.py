
from collections import defaultdict
import logging
import os
import time
import numpy as np
import torch
import bittensor as bt
from multiprocessing import Pool
from .experiment_setup import ExperimentSetup
from typing import Optional

logger = logging.getLogger("main_logger")

_archive_session = None

def get_archive_session():
    global _archive_session
    if _archive_session is None:
        _archive_session = bt.subtensor("archive")
    return _archive_session

def ensure_tensor_on_cpu(obj):
    return obj.cpu() if isinstance(obj, torch.Tensor) else obj


def check_file_corruption(file_path):
    """
    True if torch.load or torch.jit.load can read the file, else False.
    """
    try:
        torch.load(file_path, map_location="cpu")
        logger.debug(f"{file_path} loaded with torch.load")
        return True
    except Exception as e:
        logger.warning(f"{file_path} failed torch.load: {e}")
        try:
            torch.jit.load(file_path, map_location="cpu")
            logger.debug(f"{file_path} loaded with torch.jit.load")
            return True
        except Exception as e:
            logger.error(f"{file_path} failed torch.jit.load: {e}")
            return False


def download_metagraph(netuid, start_block, file_prefix, max_retries=5, retry_delay=5):
    """
    1) If file for block already exists and isn't corrupted, we skip.
    2) Otherwise, we download a new one, save, verify; if corrupted, remove and go to next block.
    """
    block = start_block
    for attempt in range(max_retries):
        file_path = f"{file_prefix}_{block}.pt"

        # 1) If file exists and is valid, skip download
        if os.path.isfile(file_path):
            if check_file_corruption(file_path):
                logger.info(f"Block {block}: File already valid, skipping.")
                return block
            else:
                logger.warning(f"Block {block}: Corrupted file found, removing.")
                os.remove(file_path)

        # 2) Download a fresh file
        try:
            logger.info(f"Block {block}: Downloading metagraph --> {file_path}")
            archive_subtensor = bt.subtensor("archive")
            meta = archive_subtensor.metagraph(netuid=netuid, block=block, lite=False)
            meta_dict = {
                "netuid": meta.netuid,
                "block": meta.block,
                "S": ensure_tensor_on_cpu(meta.S),
                "W": ensure_tensor_on_cpu(meta.W),
                "hotkeys": meta.hotkeys,
            }
            torch.save(meta_dict, file_path)

            # 3) Check corruption
            if check_file_corruption(file_path):
                logger.debug(f"Block {block}: File verified successfully.")
                return block
            else:
                logger.error(f"Block {block}: File is corrupted, removing.")
                os.remove(file_path)

        except Exception as e:
            logger.error(f"Block {block}: Error: {e}")

        # Move on to the next block if this one fails
        block += 1
        time.sleep(retry_delay)

    # If we reach here, max_retries are used up
    raise RuntimeError(
        f"Failed to download metagraph for netuid={netuid} "
        f"starting at block={start_block} after {max_retries} retries."
    )



def load_metas_from_directory(storage_path: str, epochs_num: int) -> list[torch.Tensor]:
    metas = []

    logger.info(f"Checking directory: {storage_path}")
    if not os.path.exists(storage_path):
        logger.error(f"Directory does not exist: {storage_path}")
        return metas

    files = os.listdir(storage_path)
    logger.debug(f"Found files: {files}")

    num_files = len(files)

    if epochs_num > num_files:
        logger.warning(f"Requested {epochs_num} epochs, but only {num_files} files available.")
        epochs_num = num_files

    files = files[:epochs_num]

    for filename in sorted(files):
        file_path = os.path.join(storage_path, filename)
        logger.debug(f"Processing file: {file_path}")

        try:
            data = torch.load(file_path, map_location="cpu")
            metas.append(data)
            logger.debug(f"Loaded file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(metas)} metagraphs (Requested: {epochs_num}).")
    return metas


class DownloadMetagraph:
    def __init__(self, setup=None):
        self.setup = setup if setup else ExperimentSetup()

    def run(self):
        os.makedirs(self.setup.metagraph_storage_path, exist_ok=True)
        logger.info(f"Created directory: {self.setup.metagraph_storage_path}")

        # Prepare the (netuid, block, prefix) args
        args = []
        for netuid in self.setup.netuids:
            for conceal_period in self.setup.conceal_periods:
                for dp in range(self.setup.data_points):
                    block_early = self.setup.start_block + self.setup.tempo * dp
                    prefix_early = os.path.join(
                        self.setup.metagraph_storage_path,
                        f"netuid{netuid}_block{block_early}",
                    )
                    args.append((netuid, block_early, prefix_early))

                    block_late = self.setup.start_block + self.setup.tempo * (
                        dp + conceal_period
                    )
                    prefix_late = os.path.join(
                        self.setup.metagraph_storage_path,
                        f"netuid{netuid}_block{block_late}",
                    )
                    args.append((netuid, block_late, prefix_late))

        args = list(set(args))  # remove duplicates
        logger.debug(f"Prepared download arguments: {args}")

        try:
            with Pool(processes=self.setup.processes) as pool:
                pool.starmap(download_metagraph, args)
            logger.info("DownloadMetagraph run completed.")
        except Exception as e:
            logger.error("Error occurred during metagraph download in pool.", e)
            raise

def fetch_metagraph_hotkeys(
    netuid: int,
    block: int,
    max_retries: int = 5,
    retry_delay: float = 5.0,
) -> list[str]:
    """
    Fetch the full metagraph for `netuid` at `block` and return only the
    256-length list of hotkeys (one per slot). Retries RPC up to max_retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with bt.subtensor("archive") as archive:
                meta = archive.metagraph(netuid=netuid, block=block, lite=False)
                hotkeys: list[str] = meta.hotkeys

            if len(hotkeys) != 256:
                logger.warning(
                    f"Block {block}: Expected 256 hotkeys, got {len(hotkeys)}"
                )
            return hotkeys

        except Exception as exc:
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed fetching block {block}: {exc}"
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Unable to fetch metagraph hotkeys for netuid={netuid}, "
                    f"block={block} after {max_retries} attempts."
                )


def filter_duplicate_validators(
    weight_map: dict[str, dict[str, float]],
    uids: list[int],
    stakes: dict[str, float],
    hotkeys,
) -> dict[str, dict[str, float]]:
    """
    Filters weight_map to keep only one validator per UID.
    Selection criteria: 1) Most connections, 2) If tied, larger index.
    """
    # Group validator indices by their UID
    uid_to_validator_indices = defaultdict(list)
    for idx_str in weight_map.keys():
        idx = int(idx_str)
        if 0 <= idx < len(uids):
            uid = uids[idx]
            uid_to_validator_indices[uid].append(idx)

    # Select best validator for each UID
    indices_to_keep = set()
    for uid, validator_indices in uid_to_validator_indices.items():
        if len(validator_indices) == 1:
            indices_to_keep.add(validator_indices[0])
        else:
            # Find validator with highest stake, then largest index
            best_idx = max(
                validator_indices,
                key=lambda idx: (stakes[str(idx)], idx)
            )
            indices_to_keep.add(best_idx)

    # Build filtered weight_map
    return {
        idx_str: row
        for idx_str, row in weight_map.items()
        if int(idx_str) in indices_to_keep
    }


def ordered_weights_for_uids(weight_map: dict[str, dict[str, float]], uids: list[int]) -> torch.Tensor:
    N = len(uids)
    # Use numpy array instead of nested lists
    result = np.zeros((256, 256), dtype=np.float32)

    for idx_i_str, row in weight_map.items():
        i = int(idx_i_str)
        if 0 <= i < N:
            ui = uids[i]
            if ui < 256:
                for idx_j_str, w in row.items():
                    j = int(idx_j_str)
                    if 0 <= j < N:
                        uj = uids[j]
                        if uj < 256:
                            result[ui, uj] = float(w)

    return torch.from_numpy(result)


def ordered_stakes_for_uids(
    stakes_map: dict[str, float],
    uids: list[int],
) -> torch.Tensor:
    """
    stakes_map: dict[str(idx) → stake]
    uids: list of all UIDs, where each idx_str in stakes_map references uids[idx]

    Returns a tensor S of length 256 where
        S[i] = stake for uid=i, or 0.0 if missing.
    """
    N = len(uids)

    # Use numpy array instead of list
    result = np.zeros(256, dtype=np.float32)

    for idx_str, stake in stakes_map.items():
        idx = int(idx_str)
        if 0 <= idx < N:
            uid = uids[idx]
            if uid < 256:  # bounds check
                # for duplicated uids choose highest stake
                if float(stake) >= result[uid]:
                    result[uid] = float(stake)

    return torch.from_numpy(result)


def epoch_hotkeys_by_uid(
    hotkeys: list[str],
    uids: list[int],
    stakes: dict[str, dict[str, float]],
    blocks: list[int],
    n_slots: Optional[int] = None,
) -> dict[int, list[str]]:
    """
    For each block in `blocks`, produce a list of length `n_slots` where:
      - By default slot i = "i" (string form).
      - If a given block has stakes[str(block)], then for each `"uid_idx"` in that dict:
          uid_idx = int(uid_idx_str)
          uid     = int(uids[uid_idx])
          hotkey  = hotkeys[uid_idx]
        we set slot_list[uid] = hotkey.
    If n_slots is None, scan all blocks’ stakes once to pick 256 vs. 1024:
      - If max(uid) > 255 → use n_slots=1024
      - Otherwise → n_slots=256
    Return a dict: { block_number → [hk_for_slot_0, hk_for_slot_1, …] }.
    """
    # Step 1: Auto-detect n_slots if caller passed n_slots=None
    if n_slots is None:
        max_uid_seen = -1
        for block_str, block_stakes in stakes.items():
            for uid_idx_str in block_stakes:
                uid_idx = int(uid_idx_str)
                # compute actual uid
                uid = int(uids[uid_idx])
                if uid > max_uid_seen:
                    max_uid_seen = uid
        # If any uid > 255, use 1024; otherwise 256
        n_slots = 1024 if max_uid_seen > 255 else 256

    result: dict[int, list[str]] = {}

    # Step 2: For each block, build the per-slot hotkey list
    for blk in blocks:
        # Initialize defaults: slot i → str(i)
        hk_by_slot: list[str] = [str(slot) for slot in range(n_slots)]

        block_key = str(blk)
        if block_key in stakes:
            selected_uids_stakes = {}
            for uid_idx_str, stake_amt in stakes[block_key].items():
                uid_idx = int(uid_idx_str)
                uid = int(uids[uid_idx])
                # for duplicated uids choose select one with highest stake
                if stake_amt >= selected_uids_stakes.get(uid, -1):
                    selected_uids_stakes[uid] = stake_amt
                    if 0 <= uid < n_slots:
                        hk_by_slot[uid] = hotkeys[uid_idx]
                # If uid is outside 0..n_slots−1, we simply ignore it.

        result[blk] = hk_by_slot

    return result

if __name__ == "__main__":
    DownloadMetagraph().run()
