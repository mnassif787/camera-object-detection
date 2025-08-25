import React, { useRef, useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, Square } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

const BasicObjectDetection: React.FC = () => {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);

  // Load COCO-SSD model
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Loading COCO-SSD model...');
        
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        console.log('TensorFlow.js ready');
        
        // Load the model
        const model = await cocoSsd.load();
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
        console.log('COCO-SSD model loaded successfully');
      } catch (error) {
        console.error('Error loading model:', error);
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  // Start camera
  const startCamera = async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video ready, starting detection...');
          if (videoRef.current) {
            videoRef.current.play();
            setIsDetecting(true);
            startDetection();
          }
        };
      }
    } catch (error) {
      console.error('Camera access error:', error);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsDetecting(false);
    setDetections([]);
  };

  // Main detection loop
  const startDetection = () => {
    const detect = async () => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current || !isDetecting) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) return;

      // Set canvas dimensions to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // Run object detection
        const predictions = await modelRef.current.detect(video);
        const filteredPredictions = predictions.filter(prediction => prediction.score > 0.5);
        
        // Update detections state
        setDetections(filteredPredictions);
        
        // Draw bounding boxes
        filteredPredictions.forEach(prediction => {
          const [x, y, width, height] = prediction.bbox;
          
          // Draw bounding box
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, width, height);
          
          // Draw label
          const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
          ctx.fillStyle = '#00FF00';
          ctx.font = '16px Arial';
          ctx.fillText(label, x, y > 20 ? y - 5 : y + height + 20);
        });
        
      } catch (error) {
        console.error('Detection error:', error);
      }

      // Continue loop
      if (isDetecting) {
        requestAnimationFrame(detect);
      }
    };

    // Start the detection loop
    detect();
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="p-8 text-center">
          <div className="w-12 h-12 mx-auto mb-4 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <h2 className="text-xl font-semibold mb-2">Loading Detection Model</h2>
          <p className="text-muted-foreground">Please wait while we load the AI model...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Camera Controls */}
      <div className="p-4 bg-card border-b border-border">
        <div className="flex gap-2 flex-wrap">
          {!isDetecting ? (
            <Button onClick={startCamera} disabled={!modelLoaded} className="gap-2">
              <Camera className="w-4 h-4" />
              Start Detection
            </Button>
          ) : (
            <Button onClick={stopCamera} variant="destructive" className="gap-2">
              <Square className="w-4 h-4" />
              Stop Detection
            </Button>
          )}
          
          {/* Status Info */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>Objects: {detections.length}</span>
            <span>Model: {modelLoaded ? "Ready" : "Loading"}</span>
            <span>Backend: {tf.getBackend()}</span>
          </div>
        </div>
      </div>

      {/* Camera View */}
      <div className="relative bg-black">
        <video
          ref={videoRef}
          className="w-full h-auto"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Detection Info */}
      <div className="p-4 bg-card border-t border-border">
        <h3 className="font-semibold mb-2">Detected Objects</h3>
        {detections.length === 0 ? (
          <p className="text-muted-foreground">No objects detected yet. Start the camera to begin detection.</p>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {detections.map((detection, index) => (
              <div key={index} className="p-2 bg-muted rounded text-sm">
                <span className="font-medium">{detection.class}</span>
                <span className="text-muted-foreground ml-2">
                  {Math.round(detection.score * 100)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BasicObjectDetection;