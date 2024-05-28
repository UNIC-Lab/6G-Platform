
import json
from kpi_extract import KpiExtractor

def main():
    with open("./data.json", 'r', encoding="utf-8") as f:
        data = json.load(f)

    kpi_solver = KpiExtractor()
    result = kpi_solver.kpi_extract(data["prompt"])

    with open("./data/kpi.json", 'w', encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()