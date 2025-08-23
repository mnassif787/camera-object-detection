import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, Square, Volume2, VolumeX } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

interface Detection {
  id: string;
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance: number;
  direction: string;
  lastAlertTime: number;
}

const ObjectDetectionCamera: React.FC = () => {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const animationRef = useRef<number | null>(null);
  const detectionsRef = useRef<Detection[]>([]);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [loadingStatus, setLoadingStatus] = useState('Initializing...');

  // Update ref when detections state changes
  useEffect(() => {
    detectionsRef.current = detections;
  }, [detections]);

  // Constants
  const DETECTION_INTERVAL = 100; // ms
  const FOCAL_LENGTH = 1000; // pixels
  const KNOWN_HEIGHTS: { [key: string]: number } = {
    'person': 1.7,      // 1.7 meters average height
    'car': 1.5,         // 1.5 meters average height
    'bicycle': 1.5,     // 1.5 meters average height
    'motorcycle': 1.5,  // 1.5 meters average height
    'bus': 3.0,         // 3.0 meters average height
    'truck': 3.0,       // 3.0 meters average height
    'dog': 0.5,         // 0.5 meters average height
    'cat': 0.3,         // 0.3 meters average height
    'chair': 0.9,       // 0.9 meters average height
    'table': 0.8,       // 0.8 meters average height
    'bottle': 0.25,     // 0.25 meters average height
    'cup': 0.15,        // 0.15 meters average height
    'laptop': 0.3,      // 0.3 meters average height
    'tv': 0.6,          // 0.6 meters average height
    'cell phone': 0.15, // 0.15 meters average height
    'book': 0.25,       // 0.25 meters average height
    'bowl': 0.1,        // 0.1 meters average height
    'apple': 0.08,      // 0.08 meters average height
    'banana': 0.2,      // 0.2 meters average height
    'airplane': 10.0,   // 10.0 meters average height
    'train': 4.0,       // 4.0 meters average height
    'boat': 2.0,        // 2.0 meters average height
    'fire hydrant': 0.8, // 0.8 meters average height
    'stop sign': 0.8,   // 0.8 meters average height
    'bench': 0.5,       // 0.5 meters average height
    'bird': 0.2,        // 0.2 meters average height
    'horse': 1.6,       // 1.6 meters average height
    'sheep': 1.0,       // 1.0 meters average height
    'cow': 1.4,         // 1.4 meters average height
    'elephant': 3.0,    // 3.0 meters average height
    'bear': 1.8,        // 1.8 meters average height
    'zebra': 1.4,       // 1.4 meters average height
    'giraffe': 4.0,     // 4.0 meters average height
    'backpack': 0.5,    // 0.5 meters average height
  };

  // Load COCO-SSD model with proper TensorFlow.js initialization
  useEffect(() => {
    const loadModel = async () => {
      try {
        setLoadingStatus('Initializing TensorFlow.js...');
        console.log('Initializing TensorFlow.js...');
        
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        console.log('TensorFlow.js ready');
        
        // Set backend to CPU if WebGL is not available
        setLoadingStatus('Setting up backend...');
        console.log('Setting up backend...');
        
        try {
          await tf.setBackend('webgl');
          console.log('WebGL backend set successfully');
        } catch (backendError) {
          console.log('WebGL not available, falling back to CPU');
          await tf.setBackend('cpu');
          console.log('CPU backend set successfully');
        }
        
        setLoadingStatus('Loading COCO-SSD model...');
        console.log('Loading COCO-SSD model...');
        
        const model = await cocoSsd.load();
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
        setLoadingStatus('Model loaded successfully!');
        console.log('COCO-SSD model loaded successfully');
      } catch (error) {
        console.error('Error loading model:', error);
        setLoadingStatus(`Error: ${error.message}`);
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  // Distance estimation using focal length method
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    const knownHeight = KNOWN_HEIGHTS[className] || 1.0; // Default to 1 meter if unknown
    
    // Distance = (Known Height × Focal Length) / Perceived Height
    const distance = (knownHeight * FOCAL_LENGTH) / height;
    
    // Apply mobile camera correction factor (typically 1.2-1.5x)
    const correctedDistance = distance * 1.3;
    
    return Math.max(0.5, Math.min(50, correctedDistance)); // Clamp between 0.5m and 50m
  };

  // Get direction based on object position
  const getDirection = (x: number, width: number, videoWidth: number): string => {
    const centerX = x + width / 2;
    if (centerX < videoWidth / 3) return "Left";
    if (centerX > (2 * videoWidth) / 3) return "Right";
    return "Center";
  };

  // Get color based on distance
  const getDistanceColor = (distance: number): string => {
    if (distance < 3) return "#FF0000";      // Red: < 3m
    if (distance < 5) return "#FFFF00";      // Yellow: 3-5m
    return "#00FF00";                        // Green: > 5m
  };

  // Voice synthesis with cooldown
  const speak = useCallback((message: string) => {
    if (!voiceEnabled) return;
    
    const synth = window.speechSynthesis;
    if (!synth.speaking) {
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.lang = "en-US";
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      utterance.volume = 0.8;
      synth.speak(utterance);
      console.log('Voice alert:', message);
    }
  }, [voiceEnabled]);

  // Generate voice alert message
  const generateAlertMessage = (detection: Detection): string => {
    const { class: className, distance, direction } = detection;
    
    if (distance < 3) {
      return `${className} approaching within ${distance.toFixed(1)} meters on the ${direction}`;
    } else if (distance < 5) {
      return `${className} detected ${distance.toFixed(1)} meters away on the ${direction}`;
    } else {
      return `${className} detected about ${distance.toFixed(1)} meters in the ${direction}`;
    }
  };

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          frameRate: { ideal: 30, max: 30 }
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

  // Main detection loop
  const startDetection = useCallback(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let lastDetectionTime = 0;

    const detect = async (currentTime: number) => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current || !isDetecting) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        if (isDetecting) {
          animationRef.current = requestAnimationFrame(detect);
        }
        return;
      }

      // Set canvas dimensions to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('Canvas resized to match video:', canvas.width, 'x', canvas.height);
      }

      // Update canvas CSS dimensions to match video display size
      const videoRect = video.getBoundingClientRect();
      canvas.style.width = videoRect.width + 'px';
      canvas.style.height = videoRect.height + 'px';
      canvas.style.top = '0px';
      canvas.style.left = '0px';

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Run object detection at controlled intervals
      if (currentTime - lastDetectionTime >= DETECTION_INTERVAL) {
        try {
          console.log('Running object detection...');
          
          const predictions = await modelRef.current.detect(video);
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.5);
          
          if (filteredPredictions.length > 0) {
            console.log('Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          }
          
          // Process detections with distance estimation and direction
          const newDetections: Detection[] = filteredPredictions.map(prediction => {
            const [x, y, width, height] = prediction.bbox;
            const distance = estimateDistance(prediction.bbox, prediction.class);
            const direction = getDirection(x, width, video.videoWidth);
            
            return {
              id: `${prediction.class}-${Date.now()}-${Math.random()}`,
              bbox: prediction.bbox,
              class: prediction.class,
              score: prediction.score,
              distance,
              direction,
              lastAlertTime: 0
            };
          });
          
          // Update detections state
          setDetections(newDetections);
          lastDetectionTime = currentTime;
          
          console.log(`Detection completed: ${newDetections.length} objects found`);
          
        } catch (error) {
          console.error('Detection error:', error);
        }
      }

      // Always draw current detections (from state) for smooth updates
      // Use a ref to get the latest detections without causing dependency issues
      const currentDetections = detectionsRef.current;
      if (currentDetections.length > 0) {
        console.log(`Drawing ${currentDetections.length} objects...`);
        
        currentDetections.forEach((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          const color = getDistanceColor(detection.distance);
          
          // Validate bbox coordinates
          if (x < 0 || y < 0 || x + width > canvas.width || y + height > canvas.height) {
            console.warn('Bbox coordinates out of bounds:', { x, y, width, height, canvasWidth: canvas.width, canvasHeight: canvas.height });
            return;
          }
          
          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);
          
          // Draw label background
          const label = `${detection.class} (${detection.distance.toFixed(1)}m, ${detection.direction})`;
          const labelPadding = 8;
          const labelHeight = 20;
          const labelWidth = ctx.measureText(label).width + labelPadding * 2;
          
          // Position label above the object, or below if too close to top
          const labelY = y > labelHeight + 10 ? y - 10 : y + height + 10;
          const labelX = x;
          
          // Label background
          ctx.fillStyle = color + 'CC'; // Semi-transparent
          ctx.fillRect(labelX, labelY - labelHeight + 5, labelWidth, labelHeight);
          
          // Label border
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.strokeRect(labelX, labelY - labelHeight + 5, labelWidth, labelHeight);
          
          // Label text
          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 14px Arial';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, labelX + labelPadding, labelY - labelHeight/2 + 5);
          
          // Voice alerts with cooldown
          const now = Date.now();
          if (detection.distance < 5 && now - detection.lastAlertTime > 5000) {
            const alertMessage = generateAlertMessage(detection);
            speak(alertMessage);
            detection.lastAlertTime = now;
          }
          
          console.log(`Drew object ${index}: ${detection.class} at ${detection.distance.toFixed(1)}m ${detection.direction}`);
        });
      } else {
        console.log('No objects to draw');
      }

      // Calculate FPS
      frameCount++;
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }

      // Continue loop for real-time updates
      if (isDetecting) {
        animationRef.current = requestAnimationFrame(detect);
      }
    };

    // Start the detection loop
    detect(performance.now());
  }, [speak]);

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
          <p className="text-muted-foreground">{loadingStatus}</p>
          {loadingStatus.includes('Error') && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
              <p>Try refreshing the page or check your internet connection.</p>
              <p className="mt-2">If the problem persists, the model may be temporarily unavailable.</p>
            </div>
          )}
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
          
          {/* Voice Toggle */}
          <Button
            variant={voiceEnabled ? "default" : "outline"}
            onClick={() => setVoiceEnabled(!voiceEnabled)}
            className="gap-2"
          >
            {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            {voiceEnabled ? "Voice On" : "Voice Off"}
          </Button>
          
          {/* Status Info */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>FPS: {fps}</span>
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
          className="absolute top-0 left-0 w-full h-full pointer-events-auto z-10"
          style={{ 
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'auto',
            zIndex: 10
          }}
        />
      </div>

      {/* Direction Zones Info */}
      <div className="p-4 bg-card border-t border-border">
        <h3 className="font-semibold mb-2">Direction Zones</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="text-center p-2 bg-red-50 border border-red-200 rounded">
            <div className="font-bold text-red-700">Left</div>
            <div className="text-red-600">0-33%</div>
          </div>
          <div className="text-center p-2 bg-yellow-50 border border-yellow-200 rounded">
            <div className="font-bold text-yellow-700">Center</div>
            <div className="text-yellow-600">34-66%</div>
          </div>
          <div className="text-center p-2 bg-green-50 border border-green-200 rounded">
            <div className="font-bold text-green-700">Right</div>
            <div className="text-green-600">67-100%</div>
          </div>
        </div>
      </div>

      {/* Distance Legend */}
      <div className="p-4 bg-card border-t border-border">
        <h3 className="font-semibold mb-2">Distance Colors</h3>
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Red: &lt;3m (Close)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span>Yellow: 3-5m (Medium)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>Green: &gt;5m (Far)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;