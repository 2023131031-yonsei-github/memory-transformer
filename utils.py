"""From Big Transfer(Google Research) - Should make one later."""
import collections
import json
import signal
import time
from dataclasses import dataclass

@dataclass
class LoggerArgs:
    logdir: str = "./logs"
    name: str = "NAME"

class Uninterrupt:
    """Context manager to gracefully handle interrupts.

    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """

    def __init__(self, sigs=(signal.SIGINT, signal.SIGTERM), verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            del signum  # unused
            del frame  # unused
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None


class Timer:
    """Context timing its scope."""

    def __init__(self, donecb):
        self.cb = donecb

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        t = time.time() - self.t0
        self.cb(t)


class Chrono:
    """Chronometer for poor-man's (but convenient!) profiling."""

    def __init__(self):
        self.timings = collections.OrderedDict()

    def measure(self, what):
        return Timer(lambda t: self._done(what, t))

    def _done(self, what, t):
        self.timings.setdefault(what, []).append(t)

    def times(self, what):
        return self.timings[what]

    def avgtime(self, what, dropfirst=False):
        timings = self.timings[what]
        if dropfirst and len(timings) > 1:
            timings = timings[1:]
        return sum(timings) / len(timings)

    def __str__(self, fmt="{:{w}.5f}", dropfirst=False):
        avgtimes = {k: self.avgtime(k, dropfirst) for k in self.timings}
        l = max(map(len, avgtimes))
        w = max(len(fmt.format(v, w=0)) for v in avgtimes.values())
        avg_by_time = sorted(avgtimes.items(), key=lambda t: t[1], reverse=True)
        return "\n".join(
            f"{name:{l}s}: " + fmt.format(t, w=w) + "s" for name, t in avg_by_time
        )


def setup_logger(args):
    """Creates and returns a fancy logger."""
    # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    # Why is setting up proper logging so !@?#! ugly?
    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
            },
            "handlers": {
                "stderr": {
                    "level": "INFO",
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "logfile": {
                    "level": "DEBUG",
                    "formatter": "standard",
                    "class": "logging.FileHandler",
                    "filename": os.path.join(args.logdir, args.name, "train.log"),
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["stderr", "logfile"],
                    "level": "DEBUG",
                    "propagate": True,
                },
            },
        }
    )
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    logger.info(args)
    return logger
