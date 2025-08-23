import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Button } from '@/components/ui/button';
import { Camera, Square, Volume2, VolumeX } from 'lucide-react';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

const ObjectDetectionCamera: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const animationRef = useRef<number>();

  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);

  // Initialize TensorFlow.js and load model
  useEffect(() => {
    const initializeTensorFlow = async () => {
      try {
        console.log('Starting TensorFlow.js initialization...');
        await tf.ready();
        console.log('TensorFlow.js initialized successfully');
        
        console.log('Loading COCO-SSD model...');
        const model = await cocoSsd.load();
        console.log('COCO-SSD model loaded successfully:', model);
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
        
        console.log('Model loaded and ready');
      } catch (error) {
        console.error('Error loading model:', error);
        setIsLoading(false);
      }
    };

    initializeTensorFlow();
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          frameRate: { ideal: 30, max: 30 }
        }
      });
      console.log('Camera stream obtained:', stream);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded, starting detection...');
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
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    setIsDetecting(false);
    setDetections([]);
  }, []);

  // Simple distance estimation
  const estimateDistance = (bbox: [number, number, number, number]): number => {
    const [x, y, width, height] = bbox;
    // Simple distance estimation based on object size
    const objectArea = width * height;
    const estimatedDistance = Math.max(1, Math.min(20, 1000 / objectArea));
    return Math.round(estimatedDistance * 10) / 10;
  };

  // Get direction
  const getDirection = (bbox: [number, number, number, number], canvasWidth: number): string => {
    const [x, y, width] = bbox;
    const centerX = x + width / 2;
    const threshold = canvasWidth * 0.33;
    
    if (centerX < threshold) return 'left';
    if (centerX > canvasWidth - threshold) return 'right';
    return 'center';
  };

  // Get color based on distance
  const getDistanceColor = (distance: number): string => {
    if (distance < 3) return '#FF0000'; // Red for close
    if (distance < 5) return '#FF8800'; // Orange for medium
    return '#00FF00'; // Green for far
  };

  // Simple speech
  const speakDetection = useCallback((detection: Detection) => {
    if (!speechEnabled) return;
    
    const distance = estimateDistance(detection.bbox);
    const direction = getDirection(detection.bbox, canvasRef.current?.width || 640);
    
    const message = `${detection.class} ${distance} meters ${direction}`;
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
    
    console.log('Voice alert:', message);
  }, [speechEnabled]);

  // Main detection loop
  const startDetection = useCallback(() => {
    console.log('Starting detection loop...');
    
    let lastTime = Date.now();
    let frameCount = 0;
    let lastDetectionTime = 0;
    const detectionInterval = 500; // Run detection every 500ms

    const detect = async () => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current) {
        console.log('Missing refs, stopping detection');
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const currentTime = Date.now();

      if (!ctx) {
        console.log('No canvas context');
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      // Set canvas size to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('Canvas resized to:', canvas.width, 'x', canvas.height);
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // Run object detection
        if (currentTime - lastDetectionTime >= detectionInterval) {
          console.log('Running object detection...');
          
          const predictions = await modelRef.current.detect(video);
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.3);
          
          if (filteredPredictions.length > 0) {
            console.log('Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          }
          
          const newDetections = filteredPredictions.map(prediction => ({
            bbox: prediction.bbox,
            class: prediction.class,
            score: prediction.score
          }));
          
          setDetections(newDetections);
          
          // Speak about new detections
          newDetections.forEach(detection => {
            speakDetection(detection);
          });
          
          lastDetectionTime = currentTime;
        }

        // Draw bounding boxes
        detections.forEach((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          const distance = estimateDistance(detection.bbox);
          const direction = getDirection(detection.bbox, canvas.width);
          const color = getDistanceColor(distance);
          
          console.log(`Drawing object ${index}:`, detection.class, 'at', x, y, width, height, 'distance:', distance);
          
          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);
          
          // Draw label
          const label = `${detection.class} ${distance}m ${direction}`;
          ctx.fillStyle = color;
          ctx.font = 'bold 16px Arial';
          ctx.fillText(label, x, y - 10);
          
          // Draw distance indicator
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x + width - 10, y + 10, 5, 0, 2 * Math.PI);
          ctx.fill();
        });

      } catch (error) {
        console.error('Detection error:', error);
      }

      // Calculate FPS
      frameCount++;
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }

      // Continue loop
      if (isDetecting) {
        animationRef.current = requestAnimationFrame(detect);
      }
    };

    detect();
  }, [detections, speakDetection, isDetecting]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="p-8 text-center">
          <div className="w-12 h-12 mx-auto mb-4 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <h2 className="text-xl font-semibold mb-2">Loading Detection Model</h2>
          <p className="text-muted-foreground">Initializing TensorFlow.js and COCO-SSD model...</p>
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
          
          <Button
            onClick={() => setSpeechEnabled(!speechEnabled)}
            variant="outline"
            className="gap-2"
            disabled={!window.speechSynthesis}
          >
            {speechEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            {speechEnabled ? 'Audio On' : 'Audio Off'}
          </Button>
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
          className="absolute top-0 left-0 w-full h-full pointer-events-auto z-10"
          style={{ 
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%'
          }}
        />
        
        {/* Simple Status */}
        {isDetecting && (
          <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white p-2 rounded text-xs z-20">
            <div className="font-bold mb-1">Status</div>
            <div>Objects: {detections.length}</div>
            <div>FPS: {fps}</div>
            <div>Canvas: {canvasRef.current?.width || 0} x {canvasRef.current?.height || 0}</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;