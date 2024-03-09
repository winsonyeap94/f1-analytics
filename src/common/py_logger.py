import logging

# Logger setup (avoided loguru due to conflict with joblib)
logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)


class CustomFormatter(logging.Formatter):

    grey = "\033[1;90m"
    blue = "\033[1;34m"
    black = "\033[1;30m"
    yellow = "\033[1;33m"
    red = "\033[1;31m"
    
    reset = "\033[0m"
    dt_color = "\033[0;32m"
    file_color = "\033[0;36m"
    
    format = "{dt_color}%(asctime)-23s {reset}| {level_color}%(levelname)-8s {reset}| {file_color}(%(filename)s:%(lineno)d) %(funcName)s{reset} - {level_color}%(message)s{reset}"

    FORMATS = {
        logging.DEBUG:      format.format(dt_color=dt_color, reset=reset, file_color=file_color, level_color=blue),
        logging.INFO:       format.format(dt_color=dt_color, reset=reset, file_color=file_color, level_color=black),
        logging.WARNING:    format.format(dt_color=dt_color, reset=reset, file_color=file_color, level_color=yellow),
        logging.ERROR:      format.format(dt_color=dt_color, reset=reset, file_color=file_color, level_color=red),
        logging.CRITICAL:   format.format(dt_color=dt_color, reset=reset, file_color=file_color, level_color=red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(ch)
logger.propagate = False
