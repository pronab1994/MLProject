import sys
from src.logger import logging as app_logger


def error_message_detail(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return f"Error: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        "Error occurred in python script name "
        f"[{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        app_logger.info("Divide by zero")
        raise CustomException(e)


    
        