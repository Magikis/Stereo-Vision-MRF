disparity = 16
blocksize = 5
padd = int(blocksize / 2)

for y in range(padd, img_left.shape[0] - padd - 1):
    for x in range(padd, img_left.shape[1] - padd - 1):
        block_left = img_left[y - padd: y + padd + 1, x - padd: x + padd + 1]
        for k in range(max(0, x - disparity), min(img_left.shape[1], x + disparity)):
            block_right = img_right[y - padd: y + padd + 1, k - padd:k + padd + 1]
