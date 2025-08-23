import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Camera, Square, AlertTriangle, Target, Volume2, VolumeX, Focus, Eye } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance?: number;
  direction?: 'left' | 'center' | 'right';
}

interface TrackedObject {
  id: string;
  class: string;
  bbox: [number, number, number, number];
  score: number;
  distance?: number;
  direction?: 'left' | 'center' | 'right';
  confidence: number;
  lastSeen: number;
  frameCount: number;
  stable: boolean;
  focused: boolean;
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
  const [trackedObjects, setTrackedObjects] = useState<TrackedObject[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [fps, setFps] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [focusMode, setFocusMode] = useState(false);
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

  // Object tracking and stability functions
  const calculateIoU = (bbox1: [number, number, number, number], bbox2: [number, number, number, number]): number => {
    const [x1, y1, w1, h1] = bbox1;
    const [x2, y2, w2, h2] = bbox2;
    
    const xLeft = Math.max(x1, x2);
    const yTop = Math.max(y1, y2);
    const xRight = Math.min(x1 + w1, x2 + w2);
    const yBottom = Math.min(y1 + h1, y2 + h2);
    
    if (xRight < xLeft || yBottom < yTop) return 0;
    
    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    const unionArea = w1 * h1 + w2 * h2 - intersectionArea;
    
    return intersectionArea / unionArea;
  };

  const trackObjects = (newDetections: Detection[]): TrackedObject[] => {
    const now = Date.now();
    const currentTracked = [...trackedObjects];
    const updatedTracked: TrackedObject[] = [];
    
    // Process new detections
    newDetections.forEach(detection => {
      let bestMatch: TrackedObject | null = null;
      let bestIoU = 0.3; // Minimum IoU threshold for matching
      
      // Find best matching tracked object
      currentTracked.forEach((tracked, index) => {
        if (tracked.class === detection.class) {
          const iou = calculateIoU(tracked.bbox, detection.bbox);
          if (iou > bestIoU) {
            bestIoU = iou;
            bestMatch = tracked;
          }
        }
      });
      
      if (bestMatch) {
        // Update existing tracked object
        const updated = {
          ...bestMatch,
          bbox: detection.bbox,
          score: detection.score,
          distance: detection.distance,
          direction: detection.direction,
          confidence: Math.min(1, bestMatch.confidence + 0.1),
          lastSeen: now,
          frameCount: bestMatch.frameCount + 1,
          stable: bestMatch.frameCount >= 2, // Object is stable after 2 frames instead of 3
        };
        updatedTracked.push(updated);
        
        // Remove from current tracked list
        const index = currentTracked.findIndex(t => t.id === bestMatch!.id);
        if (index > -1) currentTracked.splice(index, 1);
      } else {
        // Create new tracked object
        const newTracked: TrackedObject = {
          id: `${detection.class}-${Date.now()}-${Math.random()}`,
          class: detection.class,
          bbox: detection.bbox,
          score: detection.score,
          distance: detection.distance,
          direction: detection.direction,
          confidence: 0.5,
          lastSeen: now,
          frameCount: 1,
          stable: false,
          focused: false,
        };
        updatedTracked.push(newTracked);
      }
    });
    
    // Keep old tracked objects for a short time (for stability)
    currentTracked.forEach(tracked => {
      const timeSinceLastSeen = now - tracked.lastSeen;
      if (timeSinceLastSeen < 1000 && tracked.frameCount > 1) { // Keep for 1 second if seen multiple times
        updatedTracked.push({
          ...tracked,
          confidence: Math.max(0, tracked.confidence - 0.05), // Gradually reduce confidence
        });
      }
    });
    
    return updatedTracked;
  };

  // Focus on specific object
  const focusOnObject = useCallback((objectId: string) => {
    setTrackedObjects(prev => prev.map(obj => ({
      ...obj,
      focused: obj.id === objectId
    })));
    
    // Find the focused object
    const focused = trackedObjects.find(obj => obj.id === objectId);
    if (focused && speechEnabled) {
      const message = `Focusing on ${focused.class} ${focused.direction}, ${Math.round(focused.distance || 0)} meters away`;
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }, [trackedObjects, speechEnabled]);

  // Text-to-speech functionality using Web Speech API
  const speakDetection = useCallback((trackedObject: TrackedObject) => {
    if (!speechEnabled || !window.speechSynthesis) return;
    
    const now = Date.now();
    const objectKey = `${trackedObject.class}-${trackedObject.direction}`;
    const timeSinceLastSpoken = now - (lastSpokenTime[objectKey] || 0);
    
    // Increased cooldown to 3 seconds for more focused speech
    if (timeSinceLastSpoken < 3000) return;
    
    // Only speak about stable objects (seen for at least 3 frames)
    if (!trackedObject.stable) return;
    
    // Cancel any ongoing speech for immediate response
    window.speechSynthesis.cancel();
    
    const distance = Math.round(trackedObject.distance || 0);
    const direction = trackedObject.direction;
    
    // Create more focused speech messages
    let message = '';
    if (trackedObject.focused) {
      // Enhanced speech for focused objects
      if (distance < 2) {
        message = `Focused on ${trackedObject.class}, ${direction}, very close at ${distance} meters`;
      } else if (distance < 5) {
        message = `Focused on ${trackedObject.class}, ${direction}, ${distance} meters away`;
      } else {
        message = `Focused on ${trackedObject.class}, ${direction}, ${distance} meters away`;
      }
    } else {
      // Regular speech for stable objects
      if (distance < 2) {
        message = `${trackedObject.class}, ${direction}, very close`;
      } else if (distance < 5) {
        message = `${trackedObject.class}, ${direction}, ${distance} meters`;
      } else {
        message = `${trackedObject.class}, ${direction}, ${distance} meters`;
      }
    }
    
    // Create and configure speech for better delivery
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.0; // Normal speech rate for clarity
    utterance.pitch = 1.0;
    utterance.volume = 0.9;
    
    // Speak the message
    window.speechSynthesis.speak(utterance);
    
    // Update last spoken time with position-specific key
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

  // Main detection loop with optimized performance and object tracking
  const startDetection = useCallback(() => {
    console.log('🎯 startDetection called');
    
    let lastTime = Date.now();
    let frameCount = 0;
    let lastDetectionTime = 0;
    const detectionInterval = 500; // Increased to 500ms for more stability
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
          const predictions = await modelRef.current.detect(video);
          
          // Higher threshold for more stable detection
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.2);
          
          if (filteredPredictions.length > 0) {
            console.log('🎯 Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
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

          // Update tracked objects
          const newTrackedObjects = trackObjects(currentDetections);
          setTrackedObjects(newTrackedObjects);
          
          // Update detections state with stable objects only
          const stableDetections = newTrackedObjects
            .filter(obj => obj.stable && obj.confidence > 0.6)
            .map(obj => ({
              bbox: obj.bbox,
              class: obj.class,
              score: obj.score,
              distance: obj.distance,
              direction: obj.direction
            }));
          
          setDetections(stableDetections);
          
          // Speak about stable, high-confidence tracked objects
          newTrackedObjects.forEach(trackedObject => {
            if (trackedObject.stable && trackedObject.confidence > 0.7) {
              speakDetection(trackedObject);
            }
          });
          
          lastDetectionTime = currentTime;
        }

        // Draw tracked objects with enhanced visualization
        const objectsToDraw = focusMode 
          ? trackedObjects.filter(obj => obj.focused)
          : trackedObjects.filter(obj => obj.stable && obj.confidence > 0.3);
        
        objectsToDraw.forEach((trackedObject, index) => {
          const [x, y, width, height] = trackedObject.bbox;
          
          // Different colors for focused vs stable objects
          if (trackedObject.focused) {
            ctx.strokeStyle = '#FFD700'; // Gold for focused objects
            ctx.fillStyle = '#FFD700';
            ctx.lineWidth = 3; // Thinner lines for cleaner look
          } else {
            ctx.strokeStyle = '#00FF00'; // Green for stable objects
            ctx.fillStyle = '#00FF00';
            ctx.lineWidth = 2; // Thin lines for clean appearance
          }
          
          // Draw clean bounding box
          ctx.strokeRect(x, y, width, height);
          
          // Draw minimal object information
          const distance = Math.round(trackedObject.distance || 0);
          const confidence = Math.round(trackedObject.confidence * 100);
          const direction = trackedObject.direction;
          
          // Create clean, minimal label
          const label = `${trackedObject.class} ${distance}m ${direction}`;
          
          // Calculate label dimensions
          ctx.font = 'bold 14px Arial';
          const metrics = ctx.measureText(label);
          const labelWidth = metrics.width + 12;
          const labelHeight = 20;
          
          // Position label above the object
          let labelX = x;
          let labelY = y - labelHeight - 5;
          
          // Adjust label position if it goes off-screen
          if (labelY < 0) {
            labelY = y + height + 5; // Put label below object
          }
          if (labelX + labelWidth > canvas.width) {
            labelX = canvas.width - labelWidth - 5; // Adjust for right edge
          }
          
          // Draw clean label background
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
          
          // Draw label border
          ctx.strokeStyle = trackedObject.focused ? '#FFD700' : '#00FF00';
          ctx.lineWidth = 1;
          ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
          
          // Draw label text
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, labelX + 6, labelY + 14);
          
          // Draw subtle confidence indicator (small dot)
          const dotSize = 4;
          ctx.fillStyle = confidence >= 80 ? '#00FF00' : confidence >= 60 ? '#FFFF00' : '#FF0000';
          ctx.beginPath();
          ctx.arc(x + width - 8, y + 8, dotSize, 0, 2 * Math.PI);
          ctx.fill();
          
          // Add focus button only when not in focus mode (minimal design)
          if (!focusMode) {
            const buttonWidth = 60;
            const buttonHeight = 20;
            const buttonX = x + width - buttonWidth - 5;
            const buttonY = y + height + 5;
            
            // Clean button design
            ctx.fillStyle = 'rgba(0, 123, 255, 0.8)';
            ctx.fillRect(buttonX, buttonY, buttonWidth, buttonHeight);
            
            // Button text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 10px Arial';
            ctx.fillText('FOCUS', buttonX + 12, buttonY + 14);
          }
        });

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
  }, [speakDetection, focusMode, trackedObjects]);

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

          <Button
            onClick={() => setFocusMode(!focusMode)}
            variant={focusMode ? "default" : "outline"}
            className="gap-2"
            disabled={!isDetecting}
          >
            <Focus className="w-4 h-4" />
            {focusMode ? 'Focus Mode On' : 'Focus Mode Off'}
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
          onClick={(e) => {
            if (!focusMode) return;
            
            const canvas = canvasRef.current;
            if (!canvas) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (canvas.width / rect.width);
            const y = (e.clientY - rect.top) * (canvas.height / rect.height);
            
            // Find clicked object
            const clickedObject = trackedObjects.find(obj => {
              const [objX, objY, objWidth, objHeight] = obj.bbox;
              return x >= objX && x <= objX + objWidth && 
                     y >= objY && y <= objY + objHeight;
            });
            
            if (clickedObject) {
              focusOnObject(clickedObject.id);
            }
          }}
        />
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;