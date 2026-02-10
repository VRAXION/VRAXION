import json
lines = open(r"S:\AI\work\VRAXION_DEV\Golden Draft\probe11_telemetry.jsonl").readlines()
if not lines:
    print("NO TELEMETRY YET")
else:
    d = json.loads(lines[-1])
    print(f"Step: {d.get('step')}")
    print(f"Loss: {d.get('loss')}")
    print(f"Acc MA100: {d.get('acc_ma100')}")
    print(f"Active set: {d.get('active_set')}")
    print(f"Solo ant: {d.get('solo_ant')}")
    print(f"Gnorms: {d.get('gnorms')}")
    print(f"Total lines: {len(lines)}")
