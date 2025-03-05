import os
from gaussian_splatting import GaussianModel
from reduced_3dgs.quantization import VectorQuantizer


def quantize(source, destination, iteration, sh_degree, device, num_clusters, **kwargs):
    input = os.path.join(source, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    output = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud_quantized.ply")
    os.makedirs(os.path.join(destination, "point_cloud", "iteration_" + str(iteration)), exist_ok=True)
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(input)
    quantizer = VectorQuantizer(gaussians, num_clusters)
    quantizer.save_quantized(output)
    quantizer.load_quantized(output)
    output = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    gaussians.save_ply(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_clusters", type=int, default=256)
    args = parser.parse_args()
    quantize(**vars(args))
