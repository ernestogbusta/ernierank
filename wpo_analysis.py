from fastapi import FastAPI, HTTPException
from subprocess import Popen, PIPE, STDOUT
import json

app = FastAPI()

@app.post("/analyze_wpo")
async def analyze_wpo(url: str):
    """
    Endpoint to analyze web performance optimization using Lighthouse.
    """
    try:
        # Llama a Lighthouse desde Python
        process = Popen(['lighthouse', url, '--output=json', '--quiet', '--no-enable-error-reporting', '--chrome-flags="--headless"'], stdout=PIPE, stderr=STDOUT)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Error executing Lighthouse: {stderr.decode('utf-8')}")

        # Parsea el resultado JSON de Lighthouse
        results = json.loads(stdout)
        return {
            "url": url,
            "performance_score": results['categories']['performance']['score'],
            "details": results['categories']['performance']['auditRefs']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
