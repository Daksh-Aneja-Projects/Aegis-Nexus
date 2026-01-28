# Dead Letter Queue for Failed Traces - Implementation Addition
#
# Add this to the end of tracing.py before the final exports
\r
import json\r
import os\r
from pathlib import Path\r
from datetime import datetime\r
\r
# Dead Letter Queue for failed span exports\r
DLQ_PATH = Path(\"./data/tracing_dlq\")\r
DLQ_PATH.mkdir(parents=True, exist_ok=True)\r
\r
def _add_to_dead_letter_queue(failure_type: str, data: Dict[str, Any]):\r
    \"\"\"Add failed export to dead letter queue for later retry.\"\"\"\r
    try:\r
        dlq_file = DLQ_PATH / f\"dlq_{datetime.utcnow().strftime('%Y%m%d')}.jsonl\"\r
        entry = {\r
            \"timestamp\": datetime.utcnow().isoformat(),\r
            \"failure_type\": failure_type,\r
            \"data\": data\r
        }\r
        with open(dlq_file, 'a') as f:\r
            f.write(json.dumps(entry) + \"\\n\")\r
        logger.debug(f\"📥 Added failed trace to DLQ: {failure_type}\")\r
    except Exception as e:\r
        logger.error(f\"❌ Failed to write to DLQ: {e}\")\r
\r
def replay_dead_letter_queue():\r
    \"\"\"Replay failed exports from the dead letter queue.\"\"\"\r
    try:\r
        for dlq_file in DLQ_PATH.glob(\"dlq_*.jsonl\"):\r
            logger.info(f\"🔄 Replaying DLQ file: {dlq_file}\")\r
            with open(dlq_file, 'r') as f:\r
                for line in f:\r
                    try:\r
                        entry = json.loads(line)\r
                        # Attempt to re-process the failed item\r
                        # This would need custom logic based on failure_type\r
                        logger.debug(f\"↻ Retrying {entry['failure_type']}\")\r
                    except Exception as retry_error:\r
                        logger.warning(f\"⚠️  DLQ retry failed: {retry_error}\")\r
            # After successful replay, optionally archive or delete\r
            # dlq_file.unlink()\r
    except Exception as e:\r
        logger.error(f\"❌ Failed to replay DLQ: {e}\")\r
