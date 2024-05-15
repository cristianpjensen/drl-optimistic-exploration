import os

import pandas as pd
import neptune
from dotenv import load_dotenv


def main(id: str):
    load_dotenv()

    run = neptune.init_run(
        with_id=id,
        project=os.environ["NEPTUNE_PROJECT"],
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        mode="read-only",
    )

    agent_id = run["experiment/config/agent_id"].fetch()
    env_name = run["experiment/config/env_name"].fetch()

    print("Agent ID:", agent_id)
    print("Environment:", env_name)
    print()

    df = run["train/undiscounted_return"].fetch_values(include_timestamp=False)
    df.insert(0, "episodes", df.index + 1)

    # Account for frame skipping
    df = df.rename(columns={ "step": "frames" })
    df["frames"] = df["frames"] * 4

    print("Last 100 episode average score after k frames:")
    print("\t10M:", get_ma_100(df, 10_000_000))
    print("\t50M:", get_ma_100(df, 50_000_000))
    print("\t100M:", get_ma_100(df, 100_000_000))
    print("\t200M:", get_ma_100(df, 200_000_000))

    df["ma_100"] = df["value"].rolling(100, min_periods=0).mean()

    file = f"{agent_id}_{env_name.split('/')[-1]}".lower()

    df.to_csv(f"{file}.csv", index=False)

    print("Outputting training score data to CSV:", file)

    df_100 = df[df["episodes"] % (len(df) // 500) == 0]
    df_100.to_csv(f"{file}_100th.csv", index=False)

    print("Outputting every 100th training score data to CSV:", file)


def get_ma_100(df: pd.DataFrame, frames_seen: int) -> float:
    return df[df["frames"] <= frames_seen][-100:]["value"].mean()


if __name__ == "__main__":
    id = input("Enter the Neptune Run ID: ")
    main(id)
