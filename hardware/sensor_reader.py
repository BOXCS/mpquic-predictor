"""
hardware/sensor_reader.py — DHT11 temperature and humidity sensor reader.

Runs on the Raspberry Pi only.  Must not be imported on the server side.

Public API
──────────
    read_sensor() -> tuple[float | None, float | None]
        Return (temperature_c, humidity_pct) from the DHT11 sensor.
        Returns (None, None) if all retry attempts fail.

Configuration (sourced from hardware/config.py → .env)
────────────────────────────────────────────────────────
    DHT_GPIO_PIN        — BCM GPIO pin the sensor data line is wired to (default 4)
    DHT_SENSOR_TYPE     — "DHT11" or "DHT22" (default "DHT11")
    SENSOR_INTERVAL_SEC — read cadence used by the standalone loop (default 2.0)

Retry behaviour
───────────────
The DHT11 protocol is single-wire and timing-sensitive; read errors
(adafruit_dht.RuntimeError) are common and transient.  Up to MAX_RETRIES
attempts are made with RETRY_DELAY_SEC between each before giving up.
Persistent failures are logged at WARNING level and (None, None) is returned
so callers can decide how to handle missing data — this module never crashes.
"""

import logging
import time

import board
import adafruit_dht
from dotenv import load_dotenv

from hardware.config import DHT_GPIO_PIN, DHT_SENSOR_TYPE, SENSOR_INTERVAL_SEC

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Retry constants ───────────────────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_DELAY_SEC: float = 0.5

# ── Sensor initialisation ─────────────────────────────────────────────────────
# Map the integer BCM pin number to the corresponding board.D<n> pin object.
_PIN_MAP: dict[int, object] = {
    4: board.D4,
    17: board.D17,
    18: board.D18,
    22: board.D22,
    23: board.D23,
    24: board.D24,
    25: board.D25,
    27: board.D27,
}


def _get_board_pin(bcm_pin: int) -> object:
    """Resolve a BCM integer to a board.D<n> pin, raising ValueError if unknown."""
    try:
        return _PIN_MAP[bcm_pin]
    except KeyError:
        raise ValueError(
            f"BCM pin {bcm_pin} is not mapped in sensor_reader._PIN_MAP. "
            "Add it or set DHT_GPIO_PIN to a supported pin."
        )


def _create_sensor() -> adafruit_dht.DHT11 | adafruit_dht.DHT22:
    """Instantiate the correct adafruit_dht sensor object based on DHT_SENSOR_TYPE."""
    board_pin = _get_board_pin(DHT_GPIO_PIN)
    sensor_type = DHT_SENSOR_TYPE.upper()
    if sensor_type == "DHT11":
        return adafruit_dht.DHT11(board_pin, use_pulseio=False)
    elif sensor_type == "DHT22":
        return adafruit_dht.DHT22(board_pin, use_pulseio=False)
    else:
        raise ValueError(
            f"Unknown DHT_SENSOR_TYPE '{DHT_SENSOR_TYPE}'. "
            "Expected 'DHT11' or 'DHT22'."
        )


# Single shared sensor instance — created once at module load.
# use_pulseio=False avoids requiring root / special kernel modules on Pi OS.
_sensor = _create_sensor()


# ── Public API ─────────────────────────────────────────────────────────────────


def read_sensor() -> tuple[float | None, float | None]:
    """Read temperature (°C) and relative humidity (%) from the DHT11 sensor.

    Retries up to MAX_RETRIES times (with RETRY_DELAY_SEC between attempts)
    before giving up.  Always returns a 2-tuple; values are None on failure.

    Returns
    ───────
    (temperature_c, humidity_pct)
        Both floats on success.
        (None, None) if all retry attempts fail.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            temperature_c: float | None = _sensor.temperature
            humidity_pct: float | None = _sensor.humidity

            if temperature_c is None or humidity_pct is None:
                # adafruit_dht may return None without raising — treat as failure.
                raise RuntimeError("Sensor returned None without raising an exception.")

            logger.debug(
                "DHT11 read OK (attempt %d/%d): %.1f °C, %.1f %%RH",
                attempt,
                MAX_RETRIES,
                temperature_c,
                humidity_pct,
            )
            return temperature_c, humidity_pct

        except RuntimeError as exc:
            # RuntimeError is the expected transient read failure from adafruit_dht.
            last_error = exc
            logger.debug(
                "DHT11 read failed (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    logger.warning(
        "DHT11: all %d read attempts failed. Last error: %s",
        MAX_RETRIES,
        last_error,
    )
    return None, None


# ── Standalone verification loop ───────────────────────────────────────────────
# Run directly on the Raspberry Pi to confirm the sensor is wired correctly:
#   python3 -m hardware.sensor_reader


def _run_verification_loop() -> None:
    """Print sensor readings at SENSOR_INTERVAL_SEC cadence until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "sensor_reader: reading %s on BCM pin %d every %.1f s — press Ctrl+C to stop.",
        DHT_SENSOR_TYPE,
        DHT_GPIO_PIN,
        SENSOR_INTERVAL_SEC,
    )

    try:
        while True:
            temperature_c, humidity_pct = read_sensor()
            if temperature_c is not None and humidity_pct is not None:
                print(
                    f"  temperature = {temperature_c:.1f} °C | "
                    f"humidity = {humidity_pct:.1f} %RH"
                )
            else:
                print("  [WARN] read failed — sensor returned None, None")
            time.sleep(SENSOR_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("sensor_reader: stopped by user.")
    finally:
        _sensor.exit()


if __name__ == "__main__":
    _run_verification_loop()
