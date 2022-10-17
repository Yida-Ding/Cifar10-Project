import matplotlib.pyplot as plt
import numpy as np

def visualize(netname, ax):
    xs = np.arange(1,101)
    for i in range(1,7):
        data = np.load('./result/%s_model_%d.npz'%(netname,i))
        loss = data["data"]
        ax.plot(xs, loss, alpha=0.6, label="%s_model_%d"%(netname,i), lw=2)
    
    ax.legend()
    ax.set_title("%s Loss"%netname, fontsize=15)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_ylim(0,3)
    ax.set_xlim(0,100)

def vis_main():
    fig, axes = plt.subplots(2, 1, figsize=(10,12))
    visualize("LeNet", axes[0])
    visualize("DingNet", axes[1])
    plt.tight_layout()
    plt.savefig("./result/loss_vis.png")

def analyze_test(netname):
    data = np.load('./result/%s_test_res.npz'%netname)
    test = data["data"]
    test = ['{:.2%}'.format(n) for n in test]
    print(test)
    
if __name__=="__main__":
    # vis_main()
    analyze_test("DingNet")
    analyze_test("LeNet")



