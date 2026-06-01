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

        print("init revo2...")
        revo2 = Revo2Reader(nero)
        time.sleep(2.0)
        # nero.connect()

        while True:
            print("nero:")
            print(nero.read_all())
            print("--------------------------------------------------------------------------------")

            print("hand:")
            print(revo2.read_all())
            print("================================================================================")

            time.sleep(1.0)

    finally:
        nero.disconnect()