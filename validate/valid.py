# TODO:
#   1. 連上 SS.db，迴圈跑所有偵測，建立 llm_nearduplicates table
#   2. 取得 {project_name}, {state_a}, {state_b}
#   3. 跑 # python main_analyzer.py ../project_name/state_a ../project_name/state_b
#   4. 把結果儲存在 SS.db 的新 table llm_nearduplicates 裏


import sqlite3
import sys

# --- global variable ---

DB_CONFIG: dict[str, str] = {
    "file_name": "SS.db",
    "table_origin": "nearduplicates",
    "table_llm": "llm_nearduplicates"
}

MAX_DETECTION_TIME: int = 200



# --- function entry ---
if __name__ == "__main__":
    pass