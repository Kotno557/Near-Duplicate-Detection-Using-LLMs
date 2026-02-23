# TODO:
#   1. 連上 SS.db，迴圈跑所有偵測，建立 llm_testsubset table
#   2. 取得 {project_name}, {state_a}, {state_b}
#   3. 跑 # python main_analyzer.py ../project_name/state_a ../project_name/state_b
#   4. 把結果儲存在 SS.db 的新 table llm_testsubset 裏


import sqlite3
import main_analyzer

# --- global variable ---

DB_CONFIG: dict[str, str] = {
    "file_path": "./SS.db",
    "table_origin": "testsubset",
    "table_llm": "llm_testsubset"
}
MAX_DETECTION_COUNT: int = 10000


# --- code main entry ---
if __name__ == "__main__":
    # connect to db
    conn = sqlite3.connect(DB_CONFIG["file_path"])
    cursor = conn.cursor()

    # create llm_nearduplicates table if not exists
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {DB_CONFIG['table_llm']} (
            id INTEGER PRIMARY KEY,
            appname TEXT,
            state1 TEXT,
            state2 TEXT,
            classification TEXT,
            reasoning TEXT,
            execution_time REAL
        )
    ''')
    conn.commit()

    # get all rows in nearduplicates table
    cursor.execute(f"SELECT * FROM {DB_CONFIG['table_origin']}")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    # transform to dict
    rows_dict = [dict(zip(columns, row)) for row in rows]

    # get existing ids from llm_nearduplicates table
    cursor.execute(f"SELECT id FROM {DB_CONFIG['table_llm']}")
    existing_ids = set(row[0] for row in cursor.fetchall())

    # run main analyzer.py
    for i in range(0, min(len(rows_dict), MAX_DETECTION_COUNT)):
        # check if id already exists
        record_id = i
        if record_id in existing_ids:
            print(f"\n[{i+1}/{min(MAX_DETECTION_COUNT, len(rows_dict))}] Skipping: ID {record_id} already exists")
            continue
        
        appname: str = rows_dict[i]["appname"]
        img_a: str = f"test_images/{appname}/{rows_dict[i]["state1"]}.png"
        img_b: str = f"test_images/{appname}/{rows_dict[i]["state2"]}.png"
        
        print(f"\n[{i+1}/{min(MAX_DETECTION_COUNT, len(rows_dict))}] Processing: {appname}")
        print(f"  Image A: {img_a}")
        print(f"  Image B: {img_b}")

        try:
            # run main_analyzer.py
            result = main_analyzer.main(img_a, img_b)

            if result:
                # map classification to number
                classification_map = {
                    "Clone": 0,
                    "Near-Duplicate": 1,
                    "Distinct": 2
                }
                classification_num = classification_map.get(result['classification'], -1)

                # insert result into llm_nearduplicates table
                cursor.execute(f'''
                    INSERT INTO {DB_CONFIG['table_llm']} 
                    (id, appname, state1, state2, classification, reasoning, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record_id,
                    rows_dict[i]["appname"],
                    rows_dict[i]['state1'],
                    rows_dict[i]['state2'],
                    classification_num,
                    result['reasoning'],
                    result['execution_time']
                ))
                conn.commit()
                print(f"\033[92m[Saved]\033[0m {result['classification']} ({classification_num}) - {result['sub_type']} ({result['execution_time']:.2f}s)")
            else:
                print(f"\033[91m[Failed]\033[0m Analysis returned None")
        
        except Exception as e:
            print(f"\033[91m[Error]\033[0m {e}")
            continue
