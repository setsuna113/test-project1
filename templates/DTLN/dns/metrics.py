import numpy as np
from pesq import pesq
from pystoi.stoi import stoi
from joblib import Parallel, delayed


def SI_SDR(reference, estimation, sr=16000):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)。

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References:
        SDR- Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference**2, axis=-1, keepdims=True)

    optimal_scaling = (
        np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    )

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection**2, axis=-1) / np.sum(noise**2, axis=-1)
    return 10 * np.log10(ratio)


def STOI(ref, est, sr=16000):
    return stoi(ref, est, sr, extended=False)


def WB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "wb")


def NB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "nb")


# Only registered metric can be used.
REGISTERED_METRICS = {
    "SI_SDR": SI_SDR,
    "STOI": STOI,
    "WB_PESQ": WB_PESQ,
    "NB_PESQ": NB_PESQ,
}


def metrics_visualization(
    noisy_list,
    clean_list,
    enhanced_list,
    metrics_list,
    epoch=100,
    num_workers=10,
    mark="",
):
    """Get metrics on validation dataset by paralleling.

    Notes:
        1. You can register other metrics, but STOI and WB_PESQ metrics must be existence. These two metrics are
            used for checking if the current epoch is a "best epoch."
        2. If you want to use a new metric, you must register it in "utile.
    """
    assert (
        "STOI" in metrics_list and "WB_PESQ" in metrics_list
    ), "'STOI' and 'WB_PESQ' must be exist."

    # Check if the metric is registered in "util.metrics" file.
    for i in metrics_list:
        assert (
            i in REGISTERED_METRICS.keys()
        ), f"{i} is not registered, please check 'util.metrics' file."

    mean_score_dict = {}
    for metric_name in metrics_list:
        # print(f'Calculating: {metric_name}')
        # score_on_noisy = Parallel(n_jobs=num_workers)(
        #     delayed(REGISTERED_METRICS[metric_name])(ref, est)
        #     for ref, est in zip(clean_list, noisy_list)
        # )
        score_on_enhanced = Parallel(n_jobs=num_workers)(
            delayed(REGISTERED_METRICS[metric_name])(ref, est)
            for ref, est in zip(clean_list, enhanced_list)
        )

        # Add mean value of the metric to tensorboard
        # mean_score_on_noisy = np.mean(score_on_noisy)
        mean_score_on_enhanced = np.mean(score_on_enhanced)

        mean_score = {metric_name: mean_score_on_enhanced}
        mean_score_dict.update(mean_score)

    return mean_score_dict
