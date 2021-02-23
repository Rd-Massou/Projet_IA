import React, { useEffect } from "react";
import { useCanvas } from "./CanvasContext";
import './App.css'

export function Canvas() {

  const { clearCanvas } = useCanvas()
  const saveImage = () => {
    let jpeg = canvasRef.current.toDataURL("image/jpeg");
  }
  const sendData = (data) => {
    //envoie de donnée
    //reçu de donnée
    const id = 33;
    getData(id)
  }
  const getData = (id) => {

  }

  const {
    canvasRef,
    prepareCanvas,
    startDrawing,
    finishDrawing,
    draw,
  } = useCanvas();

  useEffect(() => {
    prepareCanvas();
  }, []);

  return (
    <div className="draw">
      <button onClick={clearCanvas}>Clear</button>
      <button onClick={saveImage}>Test</button>
      <canvas
          style={{
            borderRadius: "25px 25px",
            boxShadow: "5px 5px 5px 5px grey",
          }}
        onMouseDown={startDrawing}
        onMouseUp={finishDrawing}
        onMouseMove={draw}
        ref={canvasRef}
      />
    </div>
  );
}