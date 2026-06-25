import traceback
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv() 

from core.graph import agent_app

app = FastAPI(title="企业级闭环营销 Agent API")

class CopyRequest(BaseModel):
    topic: str
    tone: str

@app.post("/api/generate_copy")
def generate_copy(request: CopyRequest):
    try:  
        initial_state = {
            "topic": request.topic,
            "tone": request.tone,
            "iteration": 0,
            "is_pass": False
        }
        final_state = agent_app.invoke(initial_state)
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "content": final_state.get("draft", "未能生成文案"),
                "iterations_run": final_state.get("iteration", 0),
                "final_feedback": final_state.get("feedback", "无")
            }
        }
        
    except Exception as e:  
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Agent 运行出错：{str(e)}")