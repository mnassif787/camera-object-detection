import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Camera, Square, AlertTriangle, Target, Volume2, VolumeX } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance?: number;
  direction?: 'left' | 'center' | 'right';
}

interface Alert {
  id: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  timestamp: number;
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
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [fps, setFps] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [lastSpokenTime, setLastSpokenTime] = useState<{ [key: string]: number }>({});

  const { toast } = useToast();

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
        
        toast({
          title: "Model Loaded",
          description: "Object detection model ready for use",
        });
      } catch (error) {
        console.error('Error loading model:', error);
        toast({
          variant: "destructive",
          title: "Model Load Error",
          description: "Failed to load object detection model",
        });
      }
    };

    initializeTensorFlow();
  }, [toast]);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 },
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
            // Start detection after a brief delay to ensure state is updated
            setTimeout(() => {
              console.log('🚀 Starting detection after timeout...');
              startDetection();
            }, 100);
          }
        };
      }
    } catch (error) {
      console.error('Camera access error:', error);
      toast({
        variant: "destructive",
        title: "Camera Access",
        description: "Unable to access camera. Please check permissions.",
      });
    }
  }, [toast]);

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

  // Enhanced distance estimation using triangle similarity
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    
    // More accurate real-world dimensions (in meters) based on common object sizes
    const objectDimensions: { [key: string]: { width: number; height: number } } = {
      'person': { width: 0.45, height: 1.7 },
      'car': { width: 1.8, height: 1.5 },
      'bicycle': { width: 0.6, height: 1.1 },
      'motorcycle': { width: 0.8, height: 1.2 },
      'bus': { width: 2.5, height: 3.0 },
      'truck': { width: 2.5, height: 3.5 },
      'dog': { width: 0.3, height: 0.6 },
      'cat': { width: 0.2, height: 0.3 },
      'cup': { width: 0.08, height: 0.1 },
      'couch': { width: 2.0, height: 0.8 },
      'chair': { width: 0.5, height: 0.9 },
      'table': { width: 1.2, height: 0.75 },
      'bottle': { width: 0.07, height: 0.25 },
      'book': { width: 0.15, height: 0.23 },
      'laptop': { width: 0.35, height: 0.25 },
      'tv': { width: 1.0, height: 0.6 },
      'cell phone': { width: 0.07, height: 0.15 },
      'mouse': { width: 0.06, height: 0.04 },
      'keyboard': { width: 0.45, height: 0.15 },
      'bowl': { width: 0.15, height: 0.08 },
      'apple': { width: 0.08, height: 0.08 },
      'banana': { width: 0.03, height: 0.18 },
    };

    const objDims = objectDimensions[className] || { width: 0.5, height: 0.5 };
    
    // More realistic focal length for mobile cameras (typically 24-28mm equivalent)
    // This translates to ~600-700 pixels for a standard mobile camera sensor
    const focalLength = 650;
    
    // Use both width and height for better accuracy, prefer the dimension that's more reliable
    const widthDistance = (objDims.width * focalLength) / width;
    const heightDistance = (objDims.height * focalLength) / height;
    
    // For most objects, height is more reliable due to consistent orientation
    // But for some objects like bottles, cars, etc., width might be better
    const reliableHeightObjects = ['person', 'bottle', 'cup', 'chair', 'dog', 'cat'];
    const useHeight = reliableHeightObjects.includes(className);
    
    const distance = useHeight ? heightDistance : Math.min(widthDistance, heightDistance);
    
    // More realistic distance bounds with logarithmic scaling for very close/far objects
    let adjustedDistance = distance;
    if (distance < 0.5) adjustedDistance = 0.5; // Minimum 50cm
    if (distance > 50) adjustedDistance = 50; // Maximum 50m
    
    // Apply mobile camera correction factor (mobile cameras tend to overestimate distance)
    adjustedDistance *= 0.7;
    
    return Math.max(0.3, Math.min(50, adjustedDistance));
  };

  // Determine direction based on bounding box position
  const getDirection = (bbox: [number, number, number, number], canvasWidth: number): 'left' | 'center' | 'right' => {
    const [x, y, width] = bbox;
    const centerX = x + width / 2;
    const threshold = canvasWidth * 0.33;
    
    if (centerX < threshold) return 'left';
    if (centerX > canvasWidth - threshold) return 'right';
    return 'center';
  };

  // Text-to-speech functionality using Web Speech API
  const speakDetection = useCallback((detection: Detection) => {
    if (!speechEnabled || !window.speechSynthesis) return;
    
    const now = Date.now();
    const objectKey = detection.class;
    const timeSinceLastSpoken = now - (lastSpokenTime[objectKey] || 0);
    
    // Only speak once every 3 seconds per object to avoid spam
    if (timeSinceLastSpoken < 3000) return;
    
    const distance = Math.round(detection.distance || 0);
    const direction = detection.direction;
    
    // Create natural speech message
    let message = '';
    if (distance < 2) {
      message = `${detection.class} very close, ${direction} side`;
    } else if (distance < 5) {
      message = `${detection.class} nearby, ${direction} side, ${distance} meters`;
    } else {
      message = `${detection.class} detected, ${direction} side, ${distance} meters away`;
    }
    
    // Create and configure speech
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.1;
    utterance.pitch = 1.0;
    utterance.volume = 0.8;
    
    // Speak the message
    window.speechSynthesis.speak(utterance);
    
    // Update last spoken time
    setLastSpokenTime(prev => ({
      ...prev,
      [objectKey]: now
    }));
  }, [speechEnabled, lastSpokenTime]);

  // Generate spatial alerts
  const generateAlert = (detection: Detection): Alert | null => {
    const { class: className, distance, direction, score } = detection;
    
    if (score < 0.3) return null;
    
    let alertType: 'warning' | 'danger' | 'info' = 'info';
    let message = '';

    if (distance && distance < 10) {
      alertType = 'danger';
      message = `${className} very close on the ${direction} (~${Math.round(distance)}m)`;
    } else if (distance && distance < 25) {
      alertType = 'warning';
      message = `${className} approaching from ${direction} (~${Math.round(distance)}m)`;
    } else {
      message = `${className} detected on the ${direction} (~${Math.round(distance || 50)}m)`;
    }

    return {
      id: `${className}-${Date.now()}`,
      message,
      type: alertType,
      timestamp: Date.now()
    };
  };

  // Main detection loop with optimized performance
  const startDetection = useCallback(() => {
    console.log('🎯 startDetection called');
    
    // Don't check isDetecting here - let the detection loop handle it

    let lastTime = Date.now();
    let frameCount = 0;
    let lastDetectionTime = 0;
    const detectionInterval = 200; // Slower for debugging
    let currentDetections: Detection[] = [];

    const detect = async () => {
      console.log('🔄 Detect function called');
      
      if (!videoRef.current || !canvasRef.current || !modelRef.current) {
        console.log('❌ Missing refs:', {
          video: !!videoRef.current,
          canvas: !!canvasRef.current, 
          model: !!modelRef.current,
        });
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const currentTime = Date.now();

      if (!ctx) {
        console.log('❌ No canvas context');
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.log('⏳ Waiting for video dimensions:', { 
          width: video.videoWidth, 
          height: video.videoHeight,
          readyState: video.readyState
        });
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      // Set canvas size to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('📐 Canvas resized to:', canvas.width, 'x', canvas.height);
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // Run object detection at controlled intervals
        if (currentTime - lastDetectionTime >= detectionInterval) {
          console.log('🔍 Running detection on video...');
          
          const predictions = await modelRef.current.detect(video);
          console.log('🎯 Raw predictions:', predictions.length, predictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          
          // Lower threshold for testing
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.2);
          console.log('✅ Filtered predictions:', filteredPredictions.length);
          
          if (filteredPredictions.length > 0) {
            console.log('🎉 OBJECTS DETECTED!', filteredPredictions.map(p => p.class));
          }
          
          currentDetections = filteredPredictions.map(prediction => {
            const distance = estimateDistance(prediction.bbox, prediction.class);
            const direction = getDirection(prediction.bbox, canvas.width);
            
            return {
              bbox: prediction.bbox,
              class: prediction.class,
              score: prediction.score,
              distance,
              direction
            };
          });

          // Update detections state
          setDetections(currentDetections);
          
          // Speak about new high-confidence detections
          currentDetections.forEach(detection => {
            if (detection.score > 0.4) { // Only speak for high confidence
              speakDetection(detection);
            }
          });
          
          lastDetectionTime = currentTime;
        }

        // Always draw the current detections
        if (currentDetections.length > 0) {
          console.log('🎨 Drawing', currentDetections.length, 'detections');
        }
        
        currentDetections.forEach((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          console.log(`🎨 Drawing detection ${index}:`, detection.class, 'at', [x, y, width, height]);
          
          // Bright, highly visible colors
          ctx.strokeStyle = '#00FF00'; // Bright green
          ctx.fillStyle = '#00FF00';
          ctx.lineWidth = 4; // Thicker lines
          ctx.font = 'bold 20px Arial';
          
          // Draw bounding box
          ctx.strokeRect(x, y, width, height);
          console.log(`✏️ Drew bounding box at [${x}, ${y}, ${width}, ${height}]`);
          
          // Draw label background
          const label = `${detection.class.toUpperCase()} ${Math.round(detection.score * 100)}%`;
          const metrics = ctx.measureText(label);
          const labelWidth = metrics.width + 20;
          const labelHeight = 30;
          
          // Black background for label
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);
          
          // White text for label
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, x + 10, y - 8);
          console.log(`✏️ Drew label "${label}" at [${x + 10}, ${y - 8}]`);
        });

        // Draw a test rectangle to verify canvas is working
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 3;
        ctx.strokeRect(10, 10, 100, 50);
        ctx.fillStyle = '#FF0000';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('TEST', 15, 30);

      } catch (error) {
        console.error('❌ Detection error:', error);
      }

      // Calculate FPS
      frameCount++;
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        console.log('📊 FPS:', frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }

      // Continue loop while component is still detecting
      if (videoRef.current && canvasRef.current && modelRef.current) {
        animationRef.current = requestAnimationFrame(detect);
      } else {
        console.log('🛑 Detection loop stopped - missing refs or model');
      }
    };

    console.log('🚀 Starting detection loop...');
    detect();
  }, [speakDetection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Card className="p-8 text-center">
          <Target className="w-12 h-12 mx-auto mb-4 text-primary animate-spin" />
          <h2 className="text-xl font-semibold mb-2">Loading Detection Model</h2>
          <p className="text-muted-foreground">Initializing TensorFlow.js and COCO-SSD model...</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-card border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Target className="w-6 h-6 text-primary" />
            <h1 className="text-xl font-semibold">Object Detection</h1>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="secondary" className="gap-1">
              <Square className="w-3 h-3" />
              {detections.length} objects
            </Badge>
            <Badge variant="outline">{fps} FPS</Badge>
          </div>
        </div>
      </div>

      {/* Camera Controls */}
      <div className="p-4 bg-card border-b border-border">
        <div className="flex gap-2">
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

      {/* Alert Panel */}
      {alerts.length > 0 && (
        <div className="p-4 bg-card border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-destructive" />
            <span className="font-medium">Spatial Alerts</span>
          </div>
          <div className="space-y-2">
            {alerts.slice(0, 3).map(alert => (
              <div
                key={alert.id}
                className={`p-2 rounded-lg text-sm ${
                  alert.type === 'danger' 
                    ? 'bg-destructive/10 text-destructive border border-destructive/20'
                    : alert.type === 'warning'
                    ? 'bg-accent/10 text-accent-foreground border border-accent/20'
                    : 'bg-detection-info/10 text-detection-info border border-detection-info/20'
                }`}
              >
                {alert.message}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Camera View */}
      <div className="relative flex-1">
        <video
          ref={videoRef}
          className="w-full h-auto max-h-[60vh] object-cover"
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
        />
        
        {/* Overlay Info */}
        {isDetecting && (
          <div className="absolute top-4 left-4 bg-card/90 rounded-lg p-3 border border-border backdrop-blur-sm">
            <div className="text-sm space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-detection-success rounded-full animate-pulse" />
                <span>Live Detection Active</span>
              </div>
              <div className="text-muted-foreground">
                Model: COCO-SSD | FPS: {fps}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Detection Stats */}
      {detections.length > 0 && (
        <div className="p-4 bg-card border-t border-border">
          <h3 className="font-medium mb-3">Current Detections</h3>
          <div className="grid grid-cols-1 gap-2">
            {detections.slice(0, 5).map((detection, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-muted rounded-lg">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">{detection.class}</Badge>
                  <span className="text-sm text-muted-foreground">
                    {Math.round(detection.score * 100)}% confidence
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-detection-info">
                    {detection.direction} • {Math.round(detection.distance || 0)}m
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ObjectDetectionCamera;