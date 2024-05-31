import re

class KpiExtractor:
    def __init__(self) -> None:
        self.key_lib = {
            # 'latency': ['latency', 'ms', 'milliseconds', 'second', 'seconds'],
            # 'jitter': ['jitter', 'ms', 'milliseconds', 'second', 'seconds'],
            'delay': ['second', 'seconds', 'minute', 'minutes', 'hour', 'hours', 'ms', 'milliseconds'],
            'accuracy': ['percent', '%'],
            # 'throughput': ['throughput', 'kbps', 'mbps', 'gbps', 'bps', 'bit per second', 'bits per second'],
            # 'packet_loss': ['packet loss', 'percent', '%']
        }

    def kpi_extract(self, command: str):
        extracted_kpis = {kpi: None for kpi in self.key_lib.keys()}

        for kpi, keywords in self.key_lib.items():
            for keyword in keywords:
                # 匹配关键词、值和单位，可以在关键词前后或直接后跟值和单位
                pattern = (
                    rf'([\d\.]+)\s*({self._units_for_kpi(kpi)})\s*[\w\'\-]*'  # 值和单位在前
                    # rf'|{keyword}\s*([\d\.]+)\s*({self._units_for_kpi(kpi)})'              # 关键词后直接跟值和单位
                    # rf'|{keyword}\s*[\w\'\-]*\s*([\d\.]+)\s*({self._units_for_kpi(kpi)})'  # 关键词在前
                    
                )
                matches = re.finditer(pattern, command, re.IGNORECASE)
                for match in matches:
                    # 检查是哪种匹配情况
                    if match.group(1):
                        value = match.group(1)
                        unit = match.group(2)
                    elif match.group(3):
                        value = match.group(3)
                        unit = match.group(4)
                    else:
                        value = match.group(5)
                        unit = match.group(6)
                    extracted_kpis[kpi] = f"{value} {unit}"
        
        return extracted_kpis

    def _units_for_kpi(self, kpi):
        # 返回一个正则表达式模式，匹配KPI的所有单位
        units = '|'.join(map(re.escape, self.key_lib[kpi]))
        return units

if __name__ == "__main__":
    # 示例用法
    extractor = KpiExtractor()
    command = "I want to finish this task in 3 seconds with accuracy of 89%."
    result = extractor.kpi_extract(command)
    print(result)
