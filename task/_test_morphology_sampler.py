import torch

from morphology_sampler import sample_dof6_initial_morphologies


def main():
    print("Testing morphology sampler...")

    morphs = sample_dof6_initial_morphologies(
        num_initial_samples=5,
        seed=21,
        device="cuda",
        cpu_output=False,
        as_list=False,
    )

    print("Sampler works.")
    print(f"type(morphs) = {type(morphs)}")
    print(f"morphs.shape = {morphs.shape}")
    print(f"morphs.dtype = {morphs.dtype}")
    print(f"morphs.device = {morphs.device}")

    assert isinstance(morphs, torch.Tensor), "Output should be a torch.Tensor."
    assert morphs.shape == (5, 7, 3), f"Expected shape (3, 7, 3), got {morphs.shape}."
    assert torch.isfinite(morphs).all(), "Morphologies contain NaN or Inf."

    print("\nmorphology:")
    print(morphs[0])
    print(morphs[1])
    print(morphs[2])
    print(morphs[3])
    print(morphs[4])

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
