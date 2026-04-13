import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyse():
    df = pd.read_csv("results.csv",index_col=0)
    print(df)

    df["bottleneck_type"] = df["bottleneck"].apply(
        lambda x: "device" if str(x).startswith("device") else "link"
    )

    def plot_util_vs_dev_count():
        plt.figure()
        sns.scatterplot(
            data=df,
            x="num_devices",
            y="util",
            # legend="Compute Queue"
            # hue="bottleneck_type"
        )

        plt.title("Utilization vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Utilization")
        plt.savefig("plots/utilisation_vs_devices_count.png")
        plt.show()
    
    def plot_queue_vs_dev_count():
        plt.figure()
        sns.scatterplot(
            data=df,
            x="num_devices",
            y="mean_compute_queue",
            label="Compute Queue"
            # hue="bottleneck_type"
        )
        sns.scatterplot(
            data=df,
            x="num_devices",
            y="mean_comm_queue",
            label="Communication Queue"
            # hue="bottleneck_type"
        )
        plt.legend()
        plt.title("Queue Length vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Queue Length")
        plt.savefig("plots/queue_length_vs_devices_count.png")
        plt.show()

    def plot_compute_queue_vs_dev_count():
        plt.figure()
        sns.scatterplot(
            data=df,
            x="num_devices",
            y="mean_compute_queue",
            label="Compute Queue"
            # hue="bottleneck_type"
        )
        plt.legend()
        plt.title("Compute Queue Length vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Compute Queue Length")
        plt.savefig("plots/compute_queue_length_vs_devices_count.png")
        plt.show()
    
    def plot_comm_queue_vs_dev_count():
        plt.figure()
        sns.scatterplot(
            data=df,
            x="num_devices",
            y="mean_comm_queue",
            label="Communication Queue"
            # hue="bottleneck_type"
        )
        plt.legend()
        plt.title("Communication Queue Length vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Communication Queue Length")
        plt.savefig("plots/comm_queue_length_vs_devices_count.png")
        plt.show()
    
    def plot_max_util_vs_dev_count():
        # df_max = df.groupby("num_devices")["util"].max().reset_index()
        idx = df.groupby('num_devices')['util'].idxmax()

        df_best = df.loc[idx].reset_index(drop=True)
        plt.figure()
        sns.scatterplot(
            data=df_best,
            x="num_devices",
            y="util",
            # markers='o'
            # legend="Compute Queue"
            hue="bottleneck_type"
        )
        # plt.plot(df_max["num_devices"], df_max["util"], marker='o')

        plt.title("Max Utilization vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Max Utilization")
        plt.savefig("plots/max_utilisation_vs_devices_count.png")
        plt.show()
    
    def plot_min_util_vs_dev_count():
        # df_min = df.groupby("num_devices")["util"].min().reset_index()
        idx = df.groupby('num_devices')['util'].idxmin()

        df_best = df.loc[idx].reset_index(drop=True)
        
        plt.figure()
        sns.scatterplot(
            data=df_best,
            x="num_devices",
            y="util",
            # markers='o'
            # legend="Compute Queue"
            hue="bottleneck_type"
        )
        # plt.plot(df_min["num_devices"], df_min["util"], marker='o')

        plt.title("Min Utilization vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Min Utilization")
        plt.savefig("plots/min_utilisation_vs_devices_count.png")
        plt.show()
    
    plot_util_vs_dev_count()
    plot_max_util_vs_dev_count()
    plot_min_util_vs_dev_count()
    plot_queue_vs_dev_count()
    plot_compute_queue_vs_dev_count()
    plot_comm_queue_vs_dev_count()

analyse()