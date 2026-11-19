"""The already approved P1-r2 finalization guard, without changes to input guards."""
import json
import scaling_output_identity as O


def certify_output(path, validator, out):
    with (out/'output_identity_events.jsonl').open('a') as events:
        def event(record):
            events.write(json.dumps(record,allow_nan=False)+'\n')
            events.flush()
        result=O.finalized_output_identity(path,validator,event=event)
    return result['identity'],result['details'],result['audit']
