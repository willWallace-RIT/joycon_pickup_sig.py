import asyncio
import pyautogui
from ajoycon import discover_joycons
import movemove as m
async def main():
    joycons = discover_joycons()
    if not joycons:
        print("No joy-cons found!")
        return
    async with joycons[0].connect() as joycon:
        print("Joy-Con connected! Use gyro and buttons as mouse control.")
        
        while True:
            status = joycon.status
            btns = status.buttons
            #print(btns)
            imu = status.imu
            if btns.x or btns.up == 1:
                dx = -imu.gyro.x * 0.1;
                dy = imu.gyro.y * 0.1;
               # print("dx:",dx)
                #print("dy:",dy)
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    #pyautogui.moveRel(dx,dy)
                    m.move(dx,dy)
            if btns.zl or btns.zr:
                #pyautogui.mouseDown(button="left")
                m.left_down()
            else:
                #pyautogui.mouseUp(button="left")
                m.left_up()

            if btns.l  or btns.r:
                #pyautogui.mouseDown(button="right")
                m.right_down()
            else:
                #pyautogui.mouseUp(button="right")
                m.right_up()
            await asyncio.sleep(0.01)



if __name__ == "__main__":
    asyncio.run(main())


