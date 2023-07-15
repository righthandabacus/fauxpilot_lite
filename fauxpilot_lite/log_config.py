# The uvicorn_logger is used to add timestamps

uvicorn_logger = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s|%(levelprefix)s|%(client_addr)s|"%(request_line)s"|%(status_code)s',
            "use_colors": True
        },
        "default": {
            "format": "%(asctime)s|%(levelname)s|%(name)s(%(filename)s:%(lineno)d)|%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["access"],
            # "level": "INFO",
            "propagate": False
        },
        "pilot": {
            "handlers": ["default"],
            "propagate": False
        }
    },
}
