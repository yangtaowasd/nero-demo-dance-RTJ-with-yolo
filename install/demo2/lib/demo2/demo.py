#!/usr/bin/env python3
from arm_and_revo2 import *
import time


if __name__ == "__main__":
    nero = Nero()
    nero.connect()

    try:
        print("connect nero...")
        while not nero.enable():
            time.sleep(0.01)

        time.sleep(2.0)

        revo2_cls = globals().get("Revo2Reader")
        if revo2_cls is None:
            print("Revo2Reader is not available, skip hand reader.")
            revo2 = None
        else:
            print("init revo2...")
            revo2 = revo2_cls(nero)
            time.sleep(2.0)
        # nero.connect()

        while True:
            print("nero:")
            print(nero.read_all())
            print("--------------------------------------------------------------------------------")

            if revo2 is not None:
                print("hand:")
                print(revo2.read_all())
                print("================================================================================")

            time.sleep(1.0)

    finally:
        nero.disconnect()
