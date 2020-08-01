#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 3
# Name :
# Student ID :
# Email Addr :
#

import cv2
import numpy as np
import os
import sys
import argparse


class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message

    """

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def hdr_read(filename: str) -> np.ndarray:
    """ Load a hdr image from a given path

    :param filename: path to hdr image
    :return: data: hdr image, ndarray type
    """
    data = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    assert data is not None, "File {0} not exist".format(filename)
    assert len(data.shape) == 3 and data.shape[2] == 3, "Input should be a 3-channel color hdr image"
    return data


def ldr_write(filename: str, data: np.ndarray) -> None:
    """ Store a ldr image to the given path

    :param filename: target path
    :param data: ldr image, ndarray type
    :return: status: if True, success; else, fail
    """
    return cv2.imwrite(filename, data)


def compute_luminance(input: np.ndarray) -> np.ndarray:
    """ compute the luminance of a color image

    :param input: color image
    :return: luminance: luminance intensity
    """
    luminance = 0.2126 * input[:, :, 0] + 0.7152 * input[:, :, 1] + 0.0722 * input[:, :, 2]
    return luminance


def map_luminance(input: np.ndarray, luminance: np.ndarray, new_luminance: np.ndarray) -> np.ndarray:
    """ use contrast reduced luminace to recompose color image

    :param input: hdr image
    :param luminance: original luminance
    :param new_luminance: contrast reduced luminance
    :return: output: ldr image
    """
    output = np.zeros(shape=input.shape)
    print(output.shape)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            output[i][j] = input[i][j]*new_luminance[i][j]/luminance[i][j]
    return output


def log_tonemap(input: np.ndarray) -> np.ndarray:
    """ global tone mapping with log operator

    :param input: hdr image
    :return: output: ldr image, value range [0, 1]
    """

    # write you code here
    # to be completed

    L = compute_luminance(input)
    Lmin = L[L>0].min()
    Lmax = L[L>0].max()
    l = 0.05*(Lmax-Lmin)
    D = (np.log(L+l)-np.log(Lmin+l))/(np.log(Lmax+l)-np.log(Lmin+l))
    output = map_luminance(input, L, D)

    return output


def bilateral_filter(input: np.ndarray, size: int, sigma_space: float, sigma_range: float) -> np.ndarray:#bonus
    """ local tone mapping with durand's operator (bilateral filtering)

    :param input: input image/map
    :param size: windows size for spatial filtering
    :param sigma_space: filter sigma for spatial kernel
    :param sigma_range: filter sigma for range kernel
    :return: output: filtered output
    """
    # write you code here
    # to be completed
    output = cv2.bilateralFilter(input, size, sigma_range, sigma_space)
    #output = np.array(input)
    # write you code here
    return output


def durand_tonemap(input: np.ndarray) -> np.ndarray:
    """ local tone mapping with durand's operator (bilateral filtering)

    :param input: hdr image
    :return: output: ldr image, value range [0, 1]
    """
    # write you code here
    # to be completed

    sigma_space = 0.02*min(input.shape[0],input.shape[1])
    sigma_range = 0.4
    size = 2*max(round(1.5*sigma_space),1) + 1
    print(sigma_space)
    print(sigma_range)
    print(size)
    L = compute_luminance(input)
    L1 = np.log10(L)
    bl = bilateral_filter(L1, size, sigma_space,sigma_range)
    dl = L1-bl
    Max = np.amax(bl)
    Min = np.amin(bl)
    r = np.log10(50)/(Max-Min)
    new_d = 10 ** (r*bl+dl)
    d = new_d/(10**np.amax(r*bl))
    output = map_luminance(input, L, d)
    # write you code here
    return output


# operator dictionary
op_dict = {
    "durand": durand_tonemap,
    "log": log_tonemap
}

if __name__ == "__main__":
    # read arguments
    parser = ArgParser(description='Tone Mapping')
    parser.add_argument("filename", metavar="HDRImage", type=str, help="path to the hdr image")
    parser.add_argument("--op", type=str, default="all", choices=["durand", "log", "all"],
                        help="tone mapping operators")
    args = parser.parse_args()
    # print banner
    banner = "CSCI3290, Spring 2020, Assignment 3: tone mapping"
    bar = "=" * len(banner)
    print("\n".join([bar, banner, bar]))
    # read hdr image
    image = hdr_read(args.filename)


    # define the whole process for tone mapping
    def process(op: str) -> None:
        """ perform tone mapping with the given operator

        :param op: the name of specific operator
        :return: None
        """
        operator = op_dict[op]
        # tone mapping
        result = operator(image)
        # gamma correction
        result = np.power(result, 1.0 / 2.2)
        # convert each channel to 8bit unsigned integer
        result_8bit = np.clip(result * 255, 0, 255).astype('uint8')
        # store the result
        target = "output/{filename}.{op}.png".format(filename=os.path.basename(args.filename), op=op)
        msg_success = lambda: print("Converted '{filename}' to '{target}' with {op} operator.".format(
            filename=args.filename, target=target, op=op
        ))
        msg_fail = lambda: print("Failed to write {0}".format(target))
        msg_success() if ldr_write(target, result_8bit) else msg_fail()


    if args.op == "all":
        [process(op) for op in op_dict.keys()]
    else:
        process(args.op)
