import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def analyse(output_dir = "plots/",results_csv="results.csv",results_queue_details_csv="results_queue_details.csv"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    df_queue_details = pd.read_csv(results_queue_details_csv)
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
        plt.savefig(f"{output_dir}/utilisation_vs_devices_count.png")
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
        plt.savefig(f"{output_dir}/queue_length_vs_devices_count.png")
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
        plt.savefig(f"{output_dir}/compute_queue_length_vs_devices_count.png")
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
        plt.savefig(f"{output_dir}/comm_queue_length_vs_devices_count.png")
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
        plt.savefig(f"{output_dir}/max_utilisation_vs_devices_count.png")
        plt.show()
    
    def plot_min_overall_queue_vs_dev_count():
        # df_max = df.groupby("num_devices")["util"].max().reset_index()
        idx = df.groupby('num_devices')['mean_overall_queue'].idxmin()

        df_best = df.loc[idx].reset_index(drop=True)
        print("\nBest partition per num_devices (min mean_overall_queue):")
        print(df_best[["num_devices", "mean_overall_queue", "partition", "partition_id"]])

        min_row_idx = df_best["mean_overall_queue"].idxmin()
        min_row = df_best.loc[min_row_idx]
        min_partition = min_row["partition"]
        min_devices = int(min_row["num_devices"])
        min_queue = float(min_row["mean_overall_queue"])
        min_partition_id = min_row["partition_id"]

        print("\nGlobal minimum among grouped minima:")
        print(f"num_devices={min_devices}, mean_overall_queue={min_queue:.4f}")
        print(f"partition={min_partition}")

        plt.figure()
        ax = sns.lineplot(
            data=df_best,
            x="num_devices",
            y="mean_overall_queue",
            marker="o",
            # markers='o'
            # legend="Compute Queue"
            # hue="bottleneck_type"
        )

        for _, row in df_best.iterrows():
            ax.text(
                row["num_devices"],
                row["mean_overall_queue"],
                f"{row['mean_overall_queue']:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black"
            )
        # plt.plot(df_max["num_devices"], df_max["util"], marker='o')

        plt.title(
            "Overall Queue Length vs Number of Devices\n"
            f"Min Queue Length for devices={min_devices} queue={min_queue:.4f}\n partition={min_partition}",
            fontsize=10
        )
        plt.xlabel("Number of Devices")
        plt.ylabel("Overall Queue Length")
        plt.savefig(f"{output_dir}/overall_queue_vs_devices_count.png")
        plt.show()

        return min_partition_id
    
    def plot_min_latency_vs_dev_count():
        # df_max = df.groupby("num_devices")["util"].max().reset_index()
        idx = df.groupby('num_devices')['mean_latency'].idxmin()

        df_best = df.loc[idx].reset_index(drop=True)
        print("\nBest partition per num_devices (min mean_latency):")
        print(df_best[["num_devices", "mean_latency", "partition"]])

        min_row_idx = df_best["mean_latency"].idxmin()
        min_row = df_best.loc[min_row_idx]
        min_partition = min_row["partition"]
        min_devices = int(min_row["num_devices"])
        min_mean_latency = float(min_row["mean_latency"])

        print("\nGlobal minimum among grouped minima:")
        print(f"num_devices={min_devices}, mean_latency={min_mean_latency:.4f}")
        print(f"partition={min_partition}")

        plt.figure()
        ax = sns.lineplot(
            data=df_best,
            x="num_devices",
            y="mean_latency",
            marker="o",
            # markers='o'
            # legend="Compute Queue"
            # hue="bottleneck_type"
        )

        for _, row in df_best.iterrows():
            ax.text(
                row["num_devices"],
                row["mean_latency"],
                f"{row['mean_latency']:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black"
            )
        # plt.plot(df_max["num_devices"], df_max["util"], marker='o')

        plt.title(
            "Average Task Completion Time vs Number of Devices\n"
            f"Min Average Task Completion Time for devices={min_devices} min_task_completion_time={min_mean_latency:.4f}\n partition={min_partition}",
            fontsize=10
        )
        plt.xlabel("Number of Devices")
        plt.ylabel("Average Task Completion Time")
        plt.savefig(f"{output_dir}/mean_latency_vs_devices_count.png")
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
        plt.savefig(f"{output_dir}/min_utilisation_vs_devices_count.png")
        plt.show()

    def plot_min_compute_delay_vs_dev_count():
        # df_min = df.groupby("num_devices")["util"].min().reset_index()
        idx = df.groupby('num_devices')['compute_delay'].idxmin()

        df_best = df.loc[idx].reset_index(drop=True)
        
        plt.figure()
        ax = sns.lineplot(
            data=df_best,
            x="num_devices",
            y="compute_delay",
            marker="o",
            # markers='o'
            # legend="Compute Queue"
            # hue="bottleneck_type"
        )

        for _, row in df_best.iterrows():
            ax.text(
                row["num_devices"],
                row["compute_delay"],
                f"{row['compute_delay']:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black"
            )

        plt.title("Min Compute Delay vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Min Compute Delay")
        plt.savefig(f"{output_dir}/min_compute_delay_vs_devices_count.png")
        plt.show()
    
    def plot_min_communication_delay_vs_dev_count():
        # df_min = df.groupby("num_devices")["util"].min().reset_index()
        idx = df.groupby('num_devices')['communication_delay'].idxmin()

        df_best = df.loc[idx].reset_index(drop=True)
        
        plt.figure()
        ax = sns.lineplot(
            data=df_best,
            x="num_devices",
            y="communication_delay",
            marker="o",
            # markers='o'
            # legend="Compute Queue"
            # hue="bottleneck_type"
        )

        for _, row in df_best.iterrows():
            ax.text(
                row["num_devices"],
                row["communication_delay"],
                f"{row['communication_delay']:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black"
            )

        plt.title("Min Communication Delay vs Number of Devices")
        plt.xlabel("Number of Devices")
        plt.ylabel("Min Communication Delay")
        plt.savefig(f"{output_dir}/min_communication_delay_vs_devices_count.png")
        plt.show()

    def plot_device_mean_queue_bar_for_partition(partition_id):
        df_part = df_queue_details[
            (df_queue_details["partition_id"] == partition_id)
            & (df_queue_details["entity_type"] == "device")
        ].copy()

        if df_part.empty:
            print(f"No device queue records found for partition_id={partition_id}")
            return

        df_part["entity_id_num"] = df_part["entity_id"].astype(int)
        df_part = df_part.sort_values("entity_id_num")

        plt.figure(figsize=(10, 4))
        ax = sns.barplot(
            data=df_part,
            x="entity_label",
            y="mean_queue_length",
            color="steelblue"
        )

        for i, row in df_part.reset_index(drop=True).iterrows():
            ax.text(
                i,
                row["mean_queue_length"],
                f"{row['mean_queue_length']:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        partition_text = str(df_part.iloc[0]["partition"])
        plt.title(
            "Per-Device Mean Queue Length\n"
            f"partition_id={partition_id}, partition={partition_text}",
            fontsize=10,
        )
        plt.xlabel("Device")
        plt.ylabel("Mean Queue Length")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/device_mean_queue_partition_{partition_id}.png")
        plt.show()

    def plot_link_mean_queue_bar_for_partition(partition_id):
        df_part = df_queue_details[
            (df_queue_details["partition_id"] == partition_id)
            & (df_queue_details["entity_type"] == "link")
        ].copy()

        if df_part.empty:
            print(f"No used-link queue records found for partition_id={partition_id}")
            return

        df_part = df_part.sort_values("entity_label")

        plt.figure(figsize=(14, 5))
        ax = sns.barplot(
            data=df_part,
            x="entity_label",
            y="mean_queue_length",
            color="darkorange"
        )

        for i, row in df_part.reset_index(drop=True).iterrows():
            ax.text(
                i,
                row["mean_queue_length"],
                f"{row['mean_queue_length']:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        partition_text = str(df_part.iloc[0]["partition"])
        plt.title(
            "Per-Used-Link Mean Queue Length\n"
            f"partition_id={partition_id}, partition={partition_text}",
            fontsize=10,
        )
        plt.xlabel("Link")
        plt.ylabel("Mean Queue Length")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/link_mean_queue_partition_{partition_id}.png")
        plt.show()
    
    plot_util_vs_dev_count()
    plot_max_util_vs_dev_count()
    plot_min_util_vs_dev_count()
    plot_queue_vs_dev_count()
    plot_compute_queue_vs_dev_count()
    plot_comm_queue_vs_dev_count()
    optimal_partition_id =  plot_min_overall_queue_vs_dev_count()
    plot_min_latency_vs_dev_count()

    partition_id_to_plot = optimal_partition_id
    plot_device_mean_queue_bar_for_partition(partition_id_to_plot)
    plot_link_mean_queue_bar_for_partition(partition_id_to_plot)

analyse("plots/","results.csv","results_queue_details.csv")