import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import ArrowSets


if __name__ == '__main__':
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    arrows, notArrows = ArrowSets.loadArrowSets()

    arrows_a = [i[0] for i in arrows]
    arrows_b = [i[1] for i in arrows]

    notArrows_a = [i[0] for i in notArrows]
    notArrows_b = [i[1] for i in notArrows]

    plt.plot(notArrows_a, notArrows_b, 'rx')
    plt.plot(arrows_a, arrows_b, 'gx')

    # plt.xlabel('a')
    # plt.ylabel('b')
    #
    # plt.title('AB plot')
    # plt.savefig('output/ab_plot.png', bbox_inches="tight")
    # plt.show()
    #
    # plt.hist(np.array(arrows_a).ravel(), 256, [0, 256])
    # plt.title('Hisztogram arrow A')
    # plt.xlabel('a')
    # plt.savefig('output/hisztogram_arrow_a.png', bbox_inches="tight")
    # plt.show()
    #
    # plt.hist(np.array(arrows_b).ravel(), 256, [0, 256])
    # plt.title('Hisztogram arrow B')
    # plt.xlabel('b')
    # plt.savefig('output/hisztogram_arrow_b.png', bbox_inches="tight")
    # plt.show()
    #
    # plt.hist(np.array(notArrows_a).ravel(), 256, [0, 256])
    # plt.title('Hisztogram notArrow A')
    # plt.xlabel('a')
    # plt.savefig('output/hisztogram_notArrow_a.png', bbox_inches="tight")
    # plt.show()
    #
    # plt.hist(np.array(notArrows_b).ravel(), 256, [0, 256])
    # plt.title('Hisztogram notArrow B')
    # plt.xlabel('b')
    # plt.savefig('output/hisztogram_notArrow_b.png', bbox_inches="tight")
    # plt.show()

    plt.xlabel('U')
    plt.ylabel('V')

    plt.title('UV plot')
    plt.savefig('output/uv_plot.png', bbox_inches="tight")
    plt.show()

    plt.hist(np.array(arrows_a).ravel(), 256, [0, 256])
    plt.title('Hisztogram arrow U')
    plt.xlabel('U')
    plt.savefig('output/hisztogram_arrow_u.png', bbox_inches="tight")
    plt.show()

    plt.hist(np.array(arrows_b).ravel(), 256, [0, 256])
    plt.title('Hisztogram arrow V')
    plt.xlabel('V')
    plt.savefig('output/hisztogram_arrow_v.png', bbox_inches="tight")
    plt.show()

    plt.hist(np.array(notArrows_a).ravel(), 256, [0, 256])
    plt.title('Hisztogram notArrow U')
    plt.xlabel('U')
    plt.savefig('output/hisztogram_notArrow_u.png', bbox_inches="tight")
    plt.show()

    plt.hist(np.array(notArrows_b).ravel(), 256, [0, 256])
    plt.title('Hisztogram notArrow V')
    plt.xlabel('V')
    plt.savefig('output/hisztogram_notArrow_v.png', bbox_inches="tight")
    plt.show()
