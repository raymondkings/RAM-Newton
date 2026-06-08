import torch

from morphology_sampler import sample_initial_morphologies


def main():
    print("Testing morphology sampler...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dof in (5, 6, 7):
        morphs = sample_initial_morphologies(
            num_initial_samples=5,
            dof=dof,
            seed=21,
            device=device,
            cpu_output=False,
            as_list=False,
        )

        print(f"Sampler works for DOF{dof}.")
        print(f"type(morphs) = {type(morphs)}")
        print(f"morphs.shape = {morphs.shape}")
        print(f"morphs.dtype = {morphs.dtype}")
        print(f"morphs.device = {morphs.device}")

        assert isinstance(morphs, torch.Tensor), "Output should be a torch.Tensor."
        assert morphs.shape == (5, dof + 1, 3), (
            f"Expected shape (5, {dof + 1}, 3), got {morphs.shape}."
        )
        assert torch.isfinite(morphs).all(), "Morphologies contain NaN or Inf."

        print("\nmorphology:")
        print(morphs[0])

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
