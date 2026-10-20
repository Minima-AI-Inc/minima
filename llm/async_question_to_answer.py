import json
import logging
from llm_chain import LLMChain
from async_queue import AsyncQueue
import control_flow_commands as cfc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat")

async def loop(
        questions_queue: AsyncQueue,
        response_queue: AsyncQueue,
):

    llm_chain = LLMChain()

    while True:
        data = await questions_queue.dequeue()
        data = data.replace("\n", "")

        if data == cfc.CFC_CLIENT_DISCONNECTED:
            response_queue.enqueue(
                json.dumps({
                    "reporter": "output_message",
                    "type": "disconnect_message",
                })
            )
            break

        if data == cfc.CFC_CHAT_STARTED:
            response_queue.enqueue(
                json.dumps({
                    "reporter": "output_message",
                    "type": "start_message",
                })
            )
            
        elif data == cfc.CFC_CHAT_STOPPED:
            response_queue.enqueue(
                json.dumps({
                    "reporter": "output_message",
                    "type": "stop_message",
                })
            )
            
        elif data:
            # Parse message with user_id
            try:
                msg_data = json.loads(data)
                message = msg_data.get("message", data)
                user_id = msg_data.get("user_id", "default_user")
            except (json.JSONDecodeError, TypeError):
                message = data
                user_id = "default_user"

            logger.info(f"Processing message for user: {user_id}")

            # Send processing status
            response_queue.enqueue(
                json.dumps({
                    "reporter": "output_message",
                    "type": "processing",
                    "message": "Processing your request...",
                })
            )

            # Process the query with user_id
            result = await llm_chain.ainvoke(message, user_id=user_id)

            # Send the answer
            response_queue.enqueue(
                json.dumps({
                    "reporter": "output_message",
                    "type": "answer",
                    "message": result["answer"],
                    "links": list(result["links"])
                })
            )