import torch


def lab_to_rgb(lab):
    assert lab.dim() == 4
    b, c, h, w = lab.size()
    assert c == 3
    lab_pixels = torch.reshape(lab.permute(0, 2, 3, 1), [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ]).to(lab.device)
    fxfyfz_pixels = torch.mm(
        lab_pixels + torch.tensor([16.0, 0.0, 0.0]).to(lab.device),
        lab_to_fxfyfz,
    )

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = ((fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(lab.device))
    exponential_mask = ((fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(lab.device))

    xyz_pixels = (3 * epsilon**2 *
                  (fxfyfz_pixels - 4 / 29.0)) * linear_mask + ((fxfyfz_pixels + 0.000001)**3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).to(lab.device))

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ]).to(lab.device)

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = ((rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(lab.device))
    exponential_mask = ((rgb_pixels > 0.0031308).type(torch.FloatTensor).to(lab.device))
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((
        (rgb_pixels + 0.000001)**(1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return torch.reshape(srgb_pixels, [b, h, w, c]).permute(0, 3, 1, 2)


def rgb_to_lab(srgb):
    assert srgb.dim() == 4
    b, c, h, w = srgb.size()
    assert c == 3
    srgb_pixels = torch.reshape(srgb.permute(0, 2, 3, 1), [-1, 3])

    linear_mask = ((srgb_pixels <= 0.04045).type(torch.FloatTensor).to(srgb.device))
    exponential_mask = ((srgb_pixels > 0.04045).type(torch.FloatTensor).to(srgb.device))
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055)**2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ]).to(srgb.device)

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).to(srgb.device),
    )

    epsilon = 6.0 / 29.0

    linear_mask = ((xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(srgb.device))

    exponential_mask = ((xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(srgb.device))

    fxfyfz_pixels = ((xyz_normalized_pixels / (3 * epsilon**2) + 4.0 / 29.0) * linear_mask +
                     ((xyz_normalized_pixels + 0.000001)**(1.0 / 3.0)) * exponential_mask)
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ]).to(srgb.device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).to(srgb.device)
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, [b, h, w, c]).permute(0, 3, 1, 2)


if __name__ == '__main__':
    # from skimage import data
    #
    # rgb_np = data.astronaut()
    # lab_np = color.rgb2lab(rgb_np)
    # rgb_pt = (
    #     torch.from_numpy(rgb_np / 255.0)
    #     .to(torch.float32)
    #     .unsqueeze(0)
    #     .permute(0, 3, 1, 2)
    # )
    # lab_pt = rgb2lab(rgb_pt)
    # rgb_pt_back = lab2rgb(lab_pt)
    # rgb_np_back = color.lab2rgb(lab_np)
    # print(rgb_np_back.max(), rgb_np_back.min())
    # print(np.max(np.abs(rgb_pt_back[0].permute(1, 2, 0).numpy() - rgb_np_back)))

    # from skimage import data
    #
    # rgb_np = data.astronaut()
    # lab_np = color.rgb2lab(rgb_np)
    # lab_pt = torch.from_numpy(lab_np).to(torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    # rgb_np_back = color.lab2rgb(lab_np)
    # rgb_pt_back = lab2rgb(lab_pt)
    # print(np.max(np.abs(rgb_pt_back[0].permute(1, 2, 0).numpy() - rgb_np_back)))

    pass
