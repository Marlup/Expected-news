import os
# Decorator function to handle inputs and error logging
def error_logger(
        path: str,
        file_name: str
        ):
    def decorator(target_func):
        def wrapper(*args, **kwargs):
            data, cache = target_func(*args, **kwargs)
            # If list of several status_codes'
            status_code_value = cache.get("status_code", "0")
            if isinstance(status_code_value, (list, tuple)) and status_code_value:
                write_batch_on_file(
                    path,
                    str(os.getpid()) + "_" + file_name,
                    cache.get("msgs", [])
                    )
                return data
            # If dict of 1 status_code
            if isinstance(status_code_value, dict) and status_code_value != "0":
                write_on_file(
                    path,
                    str(os.getpid()) + "_" + file_name,
                    f"{msg['status_code']};{msg['id']}\n"
                    )
            return data
        return wrapper
    return decorator

class FileManager():
    def __init__(self) -> None:
        self.files_map = {}
    def add_files(
            self, 
            path: str,
            files: list, 
            **kwargs
            ):
        for file_name in files:
            self._add_file(
                path,
                file_name, 
                **kwargs
                )
    def _add_file(
            self, 
            path: str,
            file_name: str,
            **kwargs
            ):
        if kwargs.get("open_mode", False):
            open_mode = kwargs["open_mode"]
        else:
            open_mode = "a"
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(
            path, 
            file_name
            )
        self.files_map[file_name] = open(
            full_path, 
            open_mode
            )
    def write_batch_on_file(
            self, 
            file_name: str,
            msgs: [list[dict], tuple[dict]]
            ):
        lock = multiprocessing.Lock()
        with lock:
            for msg in msgs:
                self._write_on_file(file_name, 
                                    f"{msg['status_code']};{msg['id']}\n")
    def _write_on_file(
            self, 
            file_name: str,
            msg: str
            ):
        self.files_map[file_name].write(msg)
    def close_all_files(self):
        for file in self.files_map.values():
            file.close()