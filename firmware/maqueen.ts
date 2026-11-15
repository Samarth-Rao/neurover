/**
 * neurover firmware: BBC micro:bit on a DFRobot Maqueen, driven over Bluetooth LE UART by
 * neurover (python main.py --rover) or any UART app.
 *
 * Flashing (once per change):
 *   1. https://makecode.microbit.org -> New Project -> switch the editor to JavaScript, replace
 *      everything with this file.
 *   2. Extensions (gear menu or the + under the blocks list): add "maqueen" (DFRobot) and
 *      "bluetooth". If MakeCode offers to remove "radio" to make room for bluetooth, accept.
 *   3. Gear menu -> Project Settings -> "No Pairing Required: Anyone can connect via Bluetooth".
 *      Save. (If the Mac paired with the micro:bit before, remove it in System Settings ->
 *      Bluetooth first.)
 *   4. Download, then copy the .hex onto the MICROBIT drive (or Connect device + Download).
 *   The LEDs show a cross until the laptop connects, then a tick (connected, stopped) or an arrow.
 *
 * Protocol: one ASCII line per write, e.g. "F\n".
 *   F forward, B back, L / R spin in place, S stop, P ping (answers "P\n"; leaves the motors alone),
 *   T<0-255> sets the spin speed for L / R (the app sends it on connect from config.MICROBIT_TURN_SPEED).
 *   Anything else stops the rover and shows "?".
 * Every command line feeds the watchdog: no command for 500 ms -> stop. Disconnect -> stop.
 *
 * Why it's written this way (the first version fell seconds behind a 10 Hz stream):
 *   - Only a CHANGE of command touches the motors. Repeats just feed the watchdog, so a resent F
 *     doesn't stop and restart the motors.
 *   - The Bluetooth handler never draws. basic.showArrow / showIcon hold the calling code for
 *     600 ms and showString("?") for 750 ms; MakeCode queues UART events while the handler is busy
 *     (up to 10), so at 10 lines a second every line waited behind seconds of drawing. The LEDs are
 *     drawn in their own loop, with plotImage (doesn't wait).
 *   - S is a real command (it used to fall through to the "?" branch).
 */

const SPEED = 130          // 0-255, forward / back
let turnSpeed = 60         // 0-255, L / R spin (130 spun ~180 degrees per turn command); the app can set it with "T<n>"
const WATCHDOG_MS = 500    // no command for this long -> stop

let current = "S"          // what the motors are doing: F B L R, or S / ? (stopped)
let lastCommand = 0        // input.runningTime() of the last command line
let connected = false

const QUESTION = images.createImage(`
    . # # # .
    . . . # .
    . . # # .
    . . . . .
    . . # . .
    `)

function stopMotors() {
    maqueen.motorStop(maqueen.Motors.All)
}

// Apply a command. Fast (a few I2C writes, no display), and does nothing for a repeat.
function drive(cmd: string) {
    if (cmd != "F" && cmd != "B" && cmd != "L" && cmd != "R") {
        stopMotors()       // S or anything unknown. Cheap, so always (also for a repeated S)
        current = (cmd == "S") ? "S" : "?"
        return
    }
    if (cmd == current) {
        return
    }
    current = cmd
    if (cmd == "F") {
        maqueen.motorRun(maqueen.Motors.All, maqueen.Dir.CW, SPEED)
    } else if (cmd == "B") {
        maqueen.motorRun(maqueen.Motors.All, maqueen.Dir.CCW, SPEED)
    } else if (cmd == "L") {
        maqueen.motorRun(maqueen.Motors.M1, maqueen.Dir.CCW, turnSpeed)
        maqueen.motorRun(maqueen.Motors.M2, maqueen.Dir.CW, turnSpeed)
    } else if (cmd == "R") {
        maqueen.motorRun(maqueen.Motors.M1, maqueen.Dir.CW, turnSpeed)
        maqueen.motorRun(maqueen.Motors.M2, maqueen.Dir.CCW, turnSpeed)
    }
}

function moving(): boolean {
    return current == "F" || current == "B" || current == "L" || current == "R"
}

bluetooth.startUartService()
stopMotors()

bluetooth.onBluetoothConnected(function () {
    connected = true
})

bluetooth.onBluetoothDisconnected(function () {
    connected = false
    drive("S")
})

bluetooth.onUartDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    let cmd = bluetooth.uartReadUntil(serial.delimiters(Delimiters.NewLine)).trim()
    if (cmd == "") {
        return
    }
    if (cmd == "P") {
        bluetooth.uartWriteString("P\n")   // ping: for latency checks and firmware detection
        return
    }
    if (cmd.charAt(0) == "T" && cmd.length > 1) {   // turn speed; a turn in progress picks it up at once
        turnSpeed = Math.max(0, Math.min(255, parseInt(cmd.substr(1))))
        if (current == "L" || current == "R") {
            let turning = current
            current = ""
            drive(turning)
        }
        return
    }
    lastCommand = input.runningTime()
    drive(cmd)
})

// Bench test: A = forward for one watchdog period, B = stop.
input.onButtonPressed(Button.A, function () {
    lastCommand = input.runningTime()
    drive("F")
})
input.onButtonPressed(Button.B, function () {
    drive("S")
})

// Watchdog: stop if the commands stop.
basic.forever(function () {
    if (moving() && input.runningTime() - lastCommand > WATCHDOG_MS) {
        drive("S")
    }
    basic.pause(20)
})

// LEDs: redraw only when something changed, here and never in the Bluetooth handler.
let shown = ""
basic.forever(function () {
    let want = (connected || moving()) ? current : "x"
    if (want != shown) {
        shown = want
        if (want == "F") {
            images.arrowImage(ArrowNames.North).plotImage(0)
        } else if (want == "B") {
            images.arrowImage(ArrowNames.South).plotImage(0)
        } else if (want == "L") {
            images.arrowImage(ArrowNames.West).plotImage(0)
        } else if (want == "R") {
            images.arrowImage(ArrowNames.East).plotImage(0)
        } else if (want == "?") {
            QUESTION.plotImage(0)
        } else if (want == "S") {
            images.iconImage(IconNames.Yes).plotImage(0)
        } else {
            images.iconImage(IconNames.No).plotImage(0)
        }
    }
    basic.pause(50)
})
