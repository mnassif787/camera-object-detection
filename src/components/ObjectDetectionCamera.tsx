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
  // Enhanced tracking properties
  velocity?: number; // meters per second
  lastPosition?: [number, number, number, number];
  proximityWarning?: boolean;
  riskLevel?: 'low' | 'medium' | 'high' | 'critical';
  lastDistance?: number;
  movementDirection?: 'approaching' | 'receding' | 'stationary' | 'lateral';
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
  const [enhancedTracking, setEnhancedTracking] = useState(true);
  const [proximityAlerts, setProximityAlerts] = useState(true);

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

  // Start camera with enhanced mobile optimization
  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera with enhanced mobile optimization...');
      
      // Enhanced video constraints for better mobile performance
      const constraints = {
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1280, max: 1920 }, // Higher resolution for better detection
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 30, max: 30 },
          // Mobile-specific optimizations
          aspectRatio: { ideal: 16/9 },
          // Enable advanced features if available
          advanced: [
            { exposureMode: 'continuous' },
            { focusMode: 'continuous' },
            { whiteBalanceMode: 'continuous' },
            { exposureTime: { min: 0, max: 1000 } },
            { iso: { min: 100, max: 800 } }
          ]
        }
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('Enhanced camera stream obtained:', stream);

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
        
        // Enhanced error handling for mobile devices
        videoRef.current.onerror = (error) => {
          console.error('Video playback error:', error);
          toast({
            variant: "destructive",
            title: "Video Error",
            description: "Camera stream playback failed. Please refresh and try again.",
          });
        };
      }
    } catch (error) {
      console.error('Camera access error:', error);
      
      // Enhanced error messages for different failure types
      let errorMessage = "Unable to access camera. Please check permissions.";
      if (error.name === 'NotAllowedError') {
        errorMessage = "Camera access denied. Please allow camera permissions and refresh.";
      } else if (error.name === 'NotFoundError') {
        errorMessage = "No camera found on this device.";
      } else if (error.name === 'NotSupportedError') {
        errorMessage = "Camera not supported on this device or browser.";
      } else if (error.name === 'NotReadableError') {
        errorMessage = "Camera is in use by another application.";
      }
      
      toast({
        variant: "destructive",
        title: "Camera Access Error",
        description: errorMessage,
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

  // Enhanced distance estimation using triangle similarity with improved accuracy
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    
    // Enhanced real-world dimensions (in meters) based on common object sizes
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
      'airplane': { width: 35.0, height: 12.0 },
      'train': { width: 3.0, height: 4.0 },
      'boat': { width: 2.0, height: 1.5 },
      'fire hydrant': { width: 0.3, height: 0.8 },
      'stop sign': { width: 0.3, height: 0.3 },
      'parking meter': { width: 0.2, height: 1.0 },
      'bench': { width: 1.5, height: 0.5 },
      'bird': { width: 0.15, height: 0.25 },
      'horse': { width: 1.0, height: 1.8 },
      'sheep': { width: 0.8, height: 1.2 },
      'cow': { width: 1.2, height: 1.4 },
      'elephant': { width: 2.5, height: 3.0 },
      'bear': { width: 1.5, height: 2.0 },
      'zebra': { width: 1.0, height: 1.8 },
      'giraffe': { width: 0.8, height: 4.5 },
      'backpack': { width: 0.3, height: 0.5 },
      'umbrella': { width: 0.8, height: 1.2 },
      'handbag': { width: 0.3, height: 0.4 },
      'suitcase': { width: 0.5, height: 0.3 },
      'frisbee': { width: 0.25, height: 0.25 },
      'skis': { width: 0.1, height: 1.7 },
      'snowboard': { width: 0.3, height: 1.5 },
      'sports ball': { width: 0.22, height: 0.22 },
      'kite': { width: 0.8, height: 0.6 },
      'baseball bat': { width: 0.05, height: 0.9 },
      'baseball glove': { width: 0.25, height: 0.25 },
      'skateboard': { width: 0.8, height: 0.2 },
      'surfboard': { width: 2.0, height: 0.6 },
      'tennis racket': { width: 0.3, height: 0.7 },
      'wine glass': { width: 0.08, height: 0.15 },
      'fork': { width: 0.02, height: 0.2 },
      'knife': { width: 0.02, height: 0.2 },
      'spoon': { width: 0.02, height: 0.2 },
      'sandwich': { width: 0.15, height: 0.08 },
      'orange': { width: 0.08, height: 0.08 },
      'broccoli': { width: 0.15, height: 0.2 },
      'carrot': { width: 0.03, height: 0.2 },
      'hot dog': { width: 0.15, height: 0.08 },
      'pizza': { width: 0.3, height: 0.3 },
      'donut': { width: 0.1, height: 0.1 },
      'cake': { width: 0.25, height: 0.1 },
      'bed': { width: 1.4, height: 2.0 },
      'dining table': { width: 1.8, height: 0.75 },
      'toilet': { width: 0.4, height: 0.7 },
      'remote': { width: 0.15, height: 0.05 },
      'microwave': { width: 0.5, height: 0.3 },
      'oven': { width: 0.6, height: 0.6 },
      'toaster': { width: 0.3, height: 0.25 },
      'sink': { width: 0.5, height: 0.2 },
      'refrigerator': { width: 0.8, height: 1.8 },
      'clock': { width: 0.3, height: 0.3 },
      'vase': { width: 0.15, height: 0.3 },
      'scissors': { width: 0.15, height: 0.05 },
      'teddy bear': { width: 0.3, height: 0.4 },
      'hair drier': { width: 0.2, height: 0.3 },
      'toothbrush': { width: 0.02, height: 0.2 },
    };

    const objDims = objectDimensions[className] || { width: 0.5, height: 0.5 };
    
    // Optimized focal length for mobile cameras (typically 24-28mm equivalent)
    // This translates to ~600-700 pixels for a standard mobile camera sensor
    const focalLength = 650;
    
    // Use both width and height for better accuracy, prefer the dimension that's more reliable
    const widthDistance = (objDims.width * focalLength) / width;
    const heightDistance = (objDims.height * focalLength) / height;
    
    // Enhanced object classification for distance calculation
    const reliableHeightObjects = ['person', 'bottle', 'cup', 'chair', 'dog', 'cat', 'fire hydrant', 'parking meter', 'bench', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'skis', 'baseball bat', 'surfboard', 'tennis racket', 'fork', 'knife', 'spoon', 'carrot', 'bed', 'toilet', 'refrigerator', 'vase', 'teddy bear', 'toothbrush'];
    const useHeight = reliableHeightObjects.includes(className);
    
    const distance = useHeight ? heightDistance : Math.min(widthDistance, heightDistance);
    
    // Enhanced distance bounds with logarithmic scaling for very close/far objects
    let adjustedDistance = distance;
    if (distance < 0.5) adjustedDistance = 0.5; // Minimum 50cm
    if (distance > 50) adjustedDistance = 50; // Maximum 50m
    
    // Apply mobile camera correction factor (mobile cameras tend to overestimate distance)
    adjustedDistance *= 0.7;
    
    return Math.max(0.3, Math.min(50, adjustedDistance));
  };

  // Enhanced direction detection with more precise zones
  const getDirection = (bbox: [number, number, number, number], canvasWidth: number): 'left' | 'center' | 'right' => {
    const [x, y, width] = bbox;
    const centerX = x + width / 2;
    const leftThreshold = canvasWidth * 0.33;
    const rightThreshold = canvasWidth * 0.67;
    
    if (centerX < leftThreshold) return 'left';
    if (centerX > rightThreshold) return 'right';
    return 'center';
  };

  // Enhanced color coding based on distance with better visual distinction
  const getDistanceColor = (distance: number): { stroke: string; fill: string; label: string } => {
    if (distance < 3) {
      return { 
        stroke: '#FF4444', // Bright red for very close objects
        fill: 'rgba(255, 68, 68, 0.2)', 
        label: 'DANGER' 
      };
    } else if (distance < 5) {
      return { 
        stroke: '#FF8800', // Orange for close objects
        fill: 'rgba(255, 136, 0, 0.2)', 
        label: 'CLOSE' 
      };
    } else if (distance < 10) {
      return { 
        stroke: '#FFCC00', // Yellow for medium distance
        fill: 'rgba(255, 204, 0, 0.2)', 
        label: 'MEDIUM' 
      };
    } else if (distance < 20) {
      return { 
        stroke: '#00CC00', // Green for safe distance
        fill: 'rgba(0, 204, 0, 0.2)', 
        label: 'SAFE' 
      };
    } else {
      return { 
        stroke: '#0088CC', // Blue for far objects
        fill: 'rgba(0, 136, 204, 0.2)', 
        label: 'FAR' 
      };
    }
  };

  // Enhanced object tracking and stability functions
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

  // Calculate object velocity and movement analysis
  const calculateVelocity = (currentBbox: [number, number, number, number], lastBbox: [number, number, number, number], 
                           currentDistance: number, lastDistance: number, timeDiff: number): { 
    velocity: number; 
    movementDirection: 'approaching' | 'receding' | 'stationary' | 'lateral';
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
  } => {
    if (!lastBbox || timeDiff === 0) {
      return { velocity: 0, movementDirection: 'stationary', riskLevel: 'low' };
    }

    const [x1, y1, w1, h1] = currentBbox;
    const [x2, y2, w2, h2] = lastBbox;
    
    // Calculate center points
    const centerX1 = x1 + w1 / 2;
    const centerY1 = y1 + h1 / 2;
    const centerX2 = x2 + w2 / 2;
    const centerY2 = y2 + h2 / 2;
    
    // Calculate pixel movement
    const pixelMovement = Math.sqrt(
      Math.pow(centerX1 - centerX2, 2) + Math.pow(centerY1 - centerY2, 2)
    );
    
    // Convert to real-world movement (approximate)
    const pixelToMeterRatio = currentDistance / Math.max(w1, h1);
    const realMovement = pixelMovement * pixelToMeterRatio;
    
    // Calculate velocity in m/s
    const velocity = realMovement / (timeDiff / 1000);
    
    // Determine movement direction
    let movementDirection: 'approaching' | 'receding' | 'stationary' | 'lateral';
    const distanceChange = currentDistance - lastDistance;
    
    if (Math.abs(distanceChange) < 0.5) {
      movementDirection = 'stationary';
    } else if (distanceChange < -0.5) {
      movementDirection = 'approaching';
    } else if (distanceChange > 0.5) {
      movementDirection = 'receding';
    } else {
      movementDirection = 'lateral';
    }
    
    // Calculate risk level based on velocity and distance
    let riskLevel: 'low' | 'medium' | 'high' | 'critical';
    if (currentDistance < 2 && velocity > 2) {
      riskLevel = 'critical';
    } else if (currentDistance < 3 && velocity > 1.5) {
      riskLevel = 'high';
    } else if (currentDistance < 5 && velocity > 1) {
      riskLevel = 'medium';
    } else {
      riskLevel = 'low';
    }
    
    return { velocity, movementDirection, riskLevel };
  };

  // Enhanced proximity warning system
  const checkProximityWarnings = (trackedObject: TrackedObject): boolean => {
    const distance = trackedObject.distance || 0;
    const velocity = trackedObject.velocity || 0;
    const movementDirection = trackedObject.movementDirection;
    
    // Critical proximity warnings
    if (distance < 1.5) return true; // Very close
    if (distance < 2.5 && movementDirection === 'approaching' && velocity > 1) return true; // Approaching quickly
    if (distance < 3 && velocity > 2) return true; // High velocity nearby
    
    return false;
  };

  // Generate enhanced spatial alerts with risk assessment
  const generateEnhancedAlert = (trackedObject: TrackedObject): Alert | null => {
    const distance = trackedObject.distance || 0;
    const velocity = trackedObject.velocity || 0;
    const movementDirection = trackedObject.movementDirection;
    const riskLevel = trackedObject.riskLevel;
    
    if (riskLevel === 'critical') {
      return {
        id: `critical-${trackedObject.id}`,
        message: `🚨 CRITICAL: ${trackedObject.class} at ${Math.round(distance)}m ${trackedObject.direction}! Immediate action required!`,
        type: 'danger',
        timestamp: Date.now()
      };
    } else if (riskLevel === 'high') {
      return {
        id: `high-${trackedObject.id}`,
        message: `⚠️ WARNING: ${trackedObject.class} approaching ${Math.round(distance)}m ${trackedObject.direction}. High risk!`,
        type: 'warning',
        timestamp: Date.now()
      };
    } else if (movementDirection === 'approaching' && distance < 5) {
      return {
        id: `approach-${trackedObject.id}`,
        message: `📢 ${trackedObject.class} approaching ${Math.round(distance)}m ${trackedObject.direction}. Stay alert.`,
        type: 'info',
        timestamp: Date.now()
      };
    }
    
    return null;
  };

  const trackObjects = (newDetections: Detection[]): TrackedObject[] => {
    const now = Date.now();
    const currentTracked = [...trackedObjects];
    const updatedTracked: TrackedObject[] = [];
    
    // Process new detections with enhanced matching
    newDetections.forEach(detection => {
      let bestMatch: TrackedObject | null = null;
      let bestIoU = 0.25; // Lowered threshold for better matching
      let bestDistance = Infinity;
      
      // Find best matching tracked object using both IoU and distance
      currentTracked.forEach((tracked, index) => {
        if (tracked.class === detection.class) {
          const iou = calculateIoU(tracked.bbox, detection.bbox);
          const distanceDiff = Math.abs((tracked.distance || 0) - (detection.distance || 0));
          
          // Enhanced matching: prefer high IoU but also consider distance consistency
          if (iou > bestIoU || (iou > 0.15 && distanceDiff < 2)) {
            if (iou > bestIoU || (iou === bestIoU && distanceDiff < bestDistance)) {
              bestIoU = iou;
              bestDistance = distanceDiff;
              bestMatch = tracked;
            }
          }
        }
      });
      
      if (bestMatch) {
        // Update existing tracked object with enhanced stability and movement tracking
        const timeDiff = now - bestMatch.lastSeen;
        const lastBbox = bestMatch.lastPosition || bestMatch.bbox;
        const lastDistance = bestMatch.lastDistance || bestMatch.distance || 0;
        
        // Calculate velocity and movement analysis (only if enhanced tracking is enabled)
        let movementData = { velocity: 0, movementDirection: 'stationary' as const, riskLevel: 'low' as const };
        let proximityWarning = false;
        
        if (enhancedTracking) {
          movementData = calculateVelocity(
            detection.bbox, 
            lastBbox, 
            detection.distance || 0, 
            lastDistance, 
            timeDiff
          );
          
          // Check for proximity warnings (only if proximity alerts are enabled)
          if (proximityAlerts) {
            proximityWarning = checkProximityWarnings({
              ...bestMatch,
              distance: detection.distance,
              velocity: movementData.velocity,
              movementDirection: movementData.movementDirection,
              riskLevel: movementData.riskLevel
            });
          }
        }
        
        const updated = {
          ...bestMatch,
          bbox: detection.bbox,
          score: detection.score,
          distance: detection.distance,
          direction: detection.direction,
          confidence: Math.min(1, bestMatch.confidence + 0.15), // Faster confidence building
          lastSeen: now,
          frameCount: bestMatch.frameCount + 1,
          stable: bestMatch.frameCount >= 2, // Object is stable after 2 frames
          // Enhanced tracking properties (only if enabled)
          lastPosition: enhancedTracking ? bestMatch.bbox : undefined,
          lastDistance: enhancedTracking ? bestMatch.distance : undefined,
          velocity: enhancedTracking ? movementData.velocity : undefined,
          movementDirection: enhancedTracking ? movementData.movementDirection : undefined,
          riskLevel: enhancedTracking ? movementData.riskLevel : undefined,
          proximityWarning: enhancedTracking && proximityAlerts ? proximityWarning : false
        };
        
        updatedTracked.push(updated);
        
        // Generate enhanced alerts based on movement and risk (only if both features are enabled)
        if (enhancedTracking && proximityAlerts) {
          const enhancedAlert = generateEnhancedAlert(updated);
          if (enhancedAlert) {
            setAlerts(prev => [enhancedAlert, ...prev.slice(0, 9)]); // Keep last 10 alerts
          }
        }
        
        // Remove from current tracked list
        const index = currentTracked.findIndex(t => t.id === bestMatch!.id);
        if (index > -1) currentTracked.splice(index, 1);
      } else {
        // Create new tracked object with enhanced initialization
        const newTracked: TrackedObject = {
          id: `${detection.class}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          class: detection.class,
          bbox: detection.bbox,
          score: detection.score,
          distance: detection.distance,
          direction: detection.direction,
          confidence: 0.6, // Higher initial confidence
          lastSeen: now,
          frameCount: 1,
          stable: false,
          focused: false,
          // Initialize enhanced properties (only if enabled)
          velocity: enhancedTracking ? 0 : undefined,
          movementDirection: enhancedTracking ? 'stationary' : undefined,
          riskLevel: enhancedTracking ? 'low' : undefined,
          proximityWarning: false
        };
        updatedTracked.push(newTracked);
      }
    });
    
    // Enhanced persistence logic for tracked objects
    currentTracked.forEach(tracked => {
      const timeSinceLastSeen = now - tracked.lastSeen;
      const maxPersistenceTime = tracked.frameCount > 3 ? 2000 : 1000; // Longer persistence for stable objects
      
      if (timeSinceLastSeen < maxPersistenceTime) {
        // Gradually reduce confidence but maintain object for stability
        const confidenceDecay = tracked.frameCount > 2 ? 0.02 : 0.05; // Slower decay for stable objects
        updatedTracked.push({
          ...tracked,
          confidence: Math.max(0, tracked.confidence - confidenceDecay),
          // Maintain focus state if object was focused
          focused: tracked.focused && tracked.confidence > 0.3,
        });
      }
    });
    
    // Sort by confidence and recency for better display priority
    return updatedTracked.sort((a, b) => {
      if (a.focused !== b.focused) return a.focused ? -1 : 1;
      if (a.stable !== b.stable) return a.stable ? -1 : 1;
      if (Math.abs(a.confidence - b.confidence) > 0.1) return b.confidence - a.confidence;
      return b.lastSeen - a.lastSeen;
    });
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

  // Enhanced voice alerts with movement and risk awareness
  const speakDetection = useCallback((trackedObject: TrackedObject) => {
    if (!speechEnabled) return;
    
    const now = Date.now();
    const objectKey = `${trackedObject.class}-${trackedObject.direction}-${trackedObject.movementDirection}`;
    const cooldown = 5000; // 5 second cooldown per object-direction-movement combination
    
    if (lastSpokenTime[objectKey] && (now - lastSpokenTime[objectKey]) < cooldown) {
      return;
    }
    
    const distance = Math.round(trackedObject.distance || 0);
    const direction = trackedObject.direction;
    const className = trackedObject.class;
    const velocity = trackedObject.velocity || 0;
    const movementDirection = trackedObject.movementDirection;
    const riskLevel = trackedObject.riskLevel;
    const proximityWarning = trackedObject.proximityWarning;
    
    // Enhanced alert messages with movement and risk context
    let message = '';
    
    if (proximityWarning) {
      message = `🚨 EMERGENCY! ${className} very close ${distance} meters ${direction}! Immediate action required!`;
    } else if (riskLevel === 'critical') {
      message = `🚨 CRITICAL ALERT! ${className} ${distance} meters ${direction}, moving at ${Math.round(velocity * 10) / 10} meters per second!`;
    } else if (riskLevel === 'high') {
      message = `⚠️ HIGH RISK! ${className} approaching ${distance} meters ${direction}, velocity ${Math.round(velocity * 10) / 10} m/s. Stay alert!`;
    } else if (movementDirection === 'approaching' && distance < 5) {
      message = `📢 ${className} approaching ${distance} meters ${direction} at ${Math.round(velocity * 10) / 10} m/s.`;
    } else if (movementDirection === 'receding') {
      message = `${className} moving away ${distance} meters ${direction}.`;
    } else if (movementDirection === 'lateral') {
      message = `${className} moving sideways ${distance} meters ${direction}.`;
    } else if (velocity > 1) {
      message = `${className} ${distance} meters ${direction}, moving at ${Math.round(velocity * 10) / 10} m/s.`;
    } else {
      message = `${className} ${distance} meters ${direction}.`;
    }
    
    // Enhanced speech synthesis with urgency based on risk level
    const utterance = new SpeechSynthesisUtterance(message);
    
    // Adjust speech parameters based on urgency
    if (proximityWarning || riskLevel === 'critical') {
      utterance.rate = 0.8; // Slower for critical alerts
      utterance.pitch = 1.3; // Higher pitch for urgency
      utterance.volume = 1.0; // Maximum volume for critical alerts
    } else if (riskLevel === 'high') {
      utterance.rate = 0.85;
      utterance.pitch = 1.2;
      utterance.volume = 0.9;
    } else {
      utterance.rate = 0.9; // Normal rate for regular alerts
      utterance.pitch = 1.1;
      utterance.volume = 0.8;
    }
    
    // Try to use a better voice if available
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(voice => 
      voice.lang.startsWith('en') && voice.name.includes('Google')
    ) || voices.find(voice => voice.lang.startsWith('en'));
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    
    // Cancel any ongoing speech for immediate response to critical alerts
    if (proximityWarning || riskLevel === 'critical') {
      window.speechSynthesis.cancel();
    }
    
    window.speechSynthesis.speak(utterance);
    setLastSpokenTime(prev => ({ ...prev, [objectKey]: now }));
    
    console.log('🗣️ Enhanced voice alert:', message);
  }, [speechEnabled]);

  // Main detection loop with optimized performance and object tracking
  const startDetection = useCallback(() => {
    console.log('🎯 startDetection called');
    
    let lastTime = Date.now();
    let frameCount = 0;
    let lastDetectionTime = 0;
    const detectionInterval = 200; // Optimized to 200ms for smooth 30 FPS performance
    let currentDetections: Detection[] = [];
    let isRunning = true;

    const detect = async () => {
      if (!isRunning) return;
      
      console.log('🔄 Detect function called');
      
      if (!videoRef.current || !canvasRef.current || !modelRef.current) {
        console.log('❌ Missing refs:', {
          video: !!videoRef.current,
          canvas: !!canvasRef.current, 
          model: !!modelRef.current,
        });
        if (isRunning) {
          animationRef.current = requestAnimationFrame(detect);
        }
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const currentTime = Date.now();

      if (!ctx) {
        console.log('❌ No canvas context');
        if (isRunning) {
          animationRef.current = requestAnimationFrame(detect);
        }
        return;
      }

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.log('⏳ Waiting for video dimensions:', { 
          width: video.videoWidth, 
          height: video.videoHeight,
          readyState: video.readyState
        });
        if (isRunning) {
          animationRef.current = requestAnimationFrame(detect);
        }
        return;
      }

      // Set canvas size to match video with performance optimization
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('📐 Canvas resized to:', canvas.width, 'x', canvas.height);
      }

      // Clear canvas efficiently
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Debug: Draw a test rectangle to verify canvas is working
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;
      ctx.strokeRect(10, 10, 100, 50);
      ctx.fillStyle = '#FF0000';
      ctx.font = 'bold 16px Arial';
      ctx.fillText('TEST', 15, 35);

      try {
        // Run object detection at controlled intervals with performance optimization
        if (currentTime - lastDetectionTime >= detectionInterval) {
          console.log('🔍 Running object detection...');
          
          // Performance optimization: Use a smaller input size for detection if needed
          const predictions = await modelRef.current.detect(video);
          
          // Enhanced filtering with lower confidence threshold for better object detection
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.15); // Lowered from 0.25 to 0.15
          
          if (filteredPredictions.length > 0) {
            console.log('🎯 Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          }
          
          // Batch process detections for better performance
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

          // Update tracked objects with enhanced stability
          const newTrackedObjects = trackObjects(currentDetections);
          setTrackedObjects(newTrackedObjects);
          
          // Update detections state with ALL objects for immediate display
          const allDetections = newTrackedObjects.map(obj => ({
            bbox: obj.bbox,
            class: obj.class,
            score: obj.score,
            distance: obj.distance,
            direction: obj.direction
          }));
          
          setDetections(allDetections);
          
          // Enhanced voice alerts with better cooldown management
          newTrackedObjects.forEach(trackedObject => {
            if (trackedObject.stable && trackedObject.confidence > 0.6) {
              speakDetection(trackedObject);
            }
          });
          
          lastDetectionTime = currentTime;
        }

        // Draw ALL detected objects immediately (not just stable ones) for better user experience
        const objectsToDraw = focusMode 
          ? trackedObjects.filter(obj => obj.focused)
          : trackedObjects; // Show all objects, not just stable ones
        
        console.log('🎨 Drawing objects:', objectsToDraw.length, 'objects to draw');
        
        // Performance optimization: Limit rendering complexity for very high object counts
        const maxObjectsToRender = 50; // Prevent performance issues with too many objects
        const objectsToRender = objectsToDraw.slice(0, maxObjectsToRender);
        
        if (objectsToDraw.length > maxObjectsToRender) {
          console.warn(`⚠️ Performance: Limiting rendering to ${maxObjectsToRender} objects out of ${objectsToDraw.length} detected`);
        }
        
        objectsToRender.forEach((trackedObject, index) => {
          const [x, y, width, height] = trackedObject.bbox;
          const distance = trackedObject.distance || 0;
          
          console.log(`🎨 Drawing object ${index}:`, trackedObject.class, 'at', x, y, width, height, 'distance:', distance);
          
          // Get color coding based on distance
          const colors = getDistanceColor(distance);
          
          // Set drawing styles based on focus and distance
          if (trackedObject.focused) {
            ctx.strokeStyle = '#FFD700'; // Gold for focused objects
            ctx.fillStyle = 'rgba(255, 215, 0, 0.3)';
            ctx.lineWidth = 4; // Thicker lines for focused objects
          } else {
            ctx.strokeStyle = colors.stroke;
            ctx.fillStyle = colors.fill;
            ctx.lineWidth = 3; // Standard line width
          }
          
          // Draw enhanced bounding box with fill
          ctx.fillRect(x, y, width, height);
          ctx.strokeRect(x, y, width, height);
          
          // Enhanced object information display
          const confidence = Math.round(trackedObject.confidence * 100);
          const direction = trackedObject.direction;
          const className = trackedObject.class;
          
          // Create enhanced label with distance priority and movement info
          const velocity = trackedObject.velocity || 0;
          const movementDirection = trackedObject.movementDirection;
          const riskLevel = trackedObject.riskLevel;
          const proximityWarning = trackedObject.proximityWarning;
          
          let label = `${className} ${Math.round(distance)}m ${direction}`;
          
          // Add movement and risk information to label
          if (velocity > 0.5) {
            label += ` ${Math.round(velocity * 10) / 10}m/s`;
          }
          if (movementDirection === 'approaching') {
            label += ' →';
          } else if (movementDirection === 'receding') {
            label += ' ←';
          }
          if (proximityWarning) {
            label += ' ⚠️';
          }
          
          // Calculate label dimensions
          ctx.font = 'bold 16px Arial';
          const metrics = ctx.measureText(label);
          const labelWidth = metrics.width + 16;
          const labelHeight = 24;
          
          // Position label above the object with smart positioning
          let labelX = x;
          let labelY = y - labelHeight - 8;
          
          // Adjust label position if it goes off-screen
          if (labelY < 0) {
            labelY = y + height + 8; // Put label below object
          }
          if (labelX + labelWidth > canvas.width) {
            labelX = canvas.width - labelWidth - 8; // Adjust for right edge
          }
          if (labelX < 8) {
            labelX = 8; // Adjust for left edge
          }
          
          // Draw enhanced label background with risk-based colors
          let labelBgColor = 'rgba(0, 0, 0, 0.9)';
          let labelBorderColor = colors.stroke;
          
          if (proximityWarning) {
            labelBgColor = 'rgba(255, 0, 0, 0.9)'; // Red background for warnings
            labelBorderColor = '#FFFFFF';
          } else if (riskLevel === 'high' || riskLevel === 'critical') {
            labelBgColor = 'rgba(255, 165, 0, 0.9)'; // Orange background for high risk
          }
          
          ctx.fillStyle = labelBgColor;
          ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
          
          // Draw label border with risk-based color
          ctx.strokeStyle = labelBorderColor;
          ctx.lineWidth = 2;
          ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
          
          // Draw label text with better visibility
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, labelX + 8, labelY + 16);
          
          // Draw enhanced distance priority indicator with risk level
          const dotSize = 6;
          const dotX = x + width - 12;
          const dotY = y + 12;
          
          // Color the dot based on risk level
          let dotColor = colors.stroke;
          if (riskLevel === 'critical') {
            dotColor = '#FF0000'; // Red for critical
          } else if (riskLevel === 'high') {
            dotColor = '#FF6600'; // Orange for high
          } else if (riskLevel === 'medium') {
            dotColor = '#FFCC00'; // Yellow for medium
          }
          
          ctx.fillStyle = dotColor;
          ctx.beginPath();
          ctx.arc(dotX, dotY, dotSize, 0, 2 * Math.PI);
          ctx.fill();
          
          // Add white border to dot for better visibility
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1;
          ctx.stroke();
          
          // Draw risk level indicator
          ctx.fillStyle = dotColor;
          ctx.font = 'bold 10px Arial';
          const riskText = riskLevel.toUpperCase();
          ctx.fillText(riskText, dotX - 8, dotY + 4);
          
          // Draw velocity indicator bar
          const velBarWidth = 50;
          const velBarHeight = 4;
          const velBarX = x + 8;
          const velBarY = y + height - 20;
          
          // Background bar
          ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
          ctx.fillRect(velBarX, velBarY, velBarWidth, velBarHeight);
          
          // Velocity level bar (normalized to 5 m/s max)
          const normalizedVelocity = Math.min(velocity / 5, 1);
          const velBarFillWidth = normalizedVelocity * velBarWidth;
          
          // Color based on velocity (green = slow, red = fast)
          let velBarColor = '#00FF00'; // Green for slow
          if (normalizedVelocity > 0.6) {
            velBarColor = '#FFCC00'; // Yellow for medium
          }
          if (normalizedVelocity > 0.8) {
            velBarColor = '#FF0000'; // Red for fast
          }
          
          ctx.fillStyle = velBarColor;
          ctx.fillRect(velBarX, velBarY, velBarFillWidth, velBarHeight);
          
          // Draw confidence indicator bar above velocity bar
          const confBarWidth = 40;
          const confBarHeight = 4;
          const confBarX = x + 8;
          const confBarY = y + height - 28;
          
          // Background bar
          ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
          ctx.fillRect(confBarX, confBarY, confBarWidth, confBarHeight);
          
          // Confidence level bar
          const confidenceWidth = (confidence / 100) * confBarWidth;
          ctx.fillStyle = confidence >= 80 ? '#00FF00' : confidence >= 60 ? '#FFFF00' : '#FF0000';
          ctx.fillRect(confBarX, confBarY, confidenceWidth, confBarHeight);
          
          // Add focus button only when not in focus mode (enhanced design)
          if (!focusMode) {
            const buttonWidth = 70;
            const buttonHeight = 24;
            const buttonX = x + width - buttonWidth - 8;
            const buttonY = y + height + 8;
            
            // Enhanced button design with gradient effect
            const gradient = ctx.createLinearGradient(buttonX, buttonY, buttonX, buttonY + buttonHeight);
            gradient.addColorStop(0, 'rgba(0, 123, 255, 0.9)');
            gradient.addColorStop(1, 'rgba(0, 86, 179, 0.9)');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(buttonX, buttonY, buttonWidth, buttonHeight);
            
            // Button border
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1;
            ctx.strokeRect(buttonX, buttonY, buttonWidth, buttonHeight);
            
            // Button text with icon
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 11px Arial';
            ctx.fillText('🎯 FOCUS', buttonX + 8, buttonY + 16);
          }
          
          // Draw direction indicator arrow for better spatial awareness
          const arrowSize = 12;
          const arrowX = x + width / 2;
          const arrowY = y - 20;
          
          ctx.strokeStyle = colors.stroke;
          ctx.lineWidth = 2;
          ctx.beginPath();
          
          if (direction === 'left') {
            ctx.moveTo(arrowX + arrowSize/2, arrowY);
            ctx.lineTo(arrowX - arrowSize/2, arrowY);
            ctx.lineTo(arrowX - arrowSize/2 + 4, arrowY - 4);
            ctx.moveTo(arrowX - arrowSize/2, arrowY);
            ctx.lineTo(arrowX - arrowSize/2 + 4, arrowY + 4);
          } else if (direction === 'right') {
            ctx.moveTo(arrowX - arrowSize/2, arrowY);
            ctx.lineTo(arrowX + arrowSize/2, arrowY);
            ctx.lineTo(arrowX + arrowSize/2 - 4, arrowY - 4);
            ctx.moveTo(arrowX + arrowSize/2, arrowY);
            ctx.lineTo(arrowX + arrowSize/2 - 4, arrowY + 4);
          } else { // center
            ctx.moveTo(arrowX - arrowSize/2, arrowY);
            ctx.lineTo(arrowX + arrowSize/2, arrowY);
            ctx.moveTo(arrowX, arrowY - arrowSize/2);
            ctx.lineTo(arrowX, arrowY + arrowSize/2);
          }
          
          ctx.stroke();
        });

      } catch (error) {
        console.error('❌ Detection error:', error);
      }

      // Calculate FPS with enhanced performance monitoring
      frameCount++;
      if (currentTime - lastTime >= 1000) {
        const currentFps = frameCount;
        setFps(currentFps);
        
        // Performance monitoring and optimization
        if (currentFps < 25) {
          console.warn('⚠️ Performance warning: FPS below 25. Consider reducing detection frequency.');
        } else if (currentFps >= 30) {
          console.log('✅ Performance optimal: FPS at or above 30');
        }
        
        console.log('📊 FPS:', currentFps, 'Objects:', objectsToDraw.length, 'Detection interval:', detectionInterval + 'ms');
        frameCount = 0;
        lastTime = currentTime;
      }

      // Continue loop while component is still detecting and running
      if (isRunning && videoRef.current && canvasRef.current && modelRef.current) {
        animationRef.current = requestAnimationFrame(detect);
      } else {
        console.log('🛑 Detection loop stopped - missing refs or model');
      }
    };

    console.log('🚀 Starting detection loop...');
    detect();
    
    // Return cleanup function
    return () => {
      isRunning = false;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [speakDetection, focusMode]); // Removed trackedObjects dependency to prevent recreation

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

          <Button
            onClick={() => setEnhancedTracking(!enhancedTracking)}
            variant={enhancedTracking ? "default" : "outline"}
            className="gap-2"
            disabled={!isDetecting}
          >
            <Target className="w-4 h-4" />
            {enhancedTracking ? 'Enhanced On' : 'Enhanced Off'}
          </Button>

          <Button
            onClick={() => setProximityAlerts(!proximityAlerts)}
            variant={proximityAlerts ? "default" : "outline"}
            className="gap-2"
            disabled={!isDetecting}
          >
            <AlertTriangle className="w-4 h-4" />
            {proximityAlerts ? 'Alerts On' : 'Alerts Off'}
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
          style={{ border: '2px solid red' }} // Debug: Add red border to see canvas
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
        
        {/* Debug Info Overlay */}
        {isDetecting && (
          <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white p-3 rounded text-xs z-20 max-w-64">
            <div className="font-bold mb-2 border-b border-white/20 pb-1">📊 Live Status</div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span>Canvas:</span>
                <span>{canvasRef.current?.width || 0} x {canvasRef.current?.height || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Video:</span>
                <span>{videoRef.current?.videoWidth || 0} x {videoRef.current?.videoHeight || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Objects:</span>
                <span className="text-green-400 font-bold">{trackedObjects.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Stable Objects:</span>
                <span>{trackedObjects.filter(obj => obj.stable).length}</span>
              </div>
              <div className="flex justify-between">
                <span>FPS:</span>
                <span className={fps >= 30 ? 'text-green-400' : fps >= 25 ? 'text-yellow-400' : 'text-red-400'}>
                  {fps}
                </span>
              </div>
            </div>
            
            {/* Object Detection Summary */}
            {trackedObjects.length > 0 && (
              <div className="mt-3 pt-2 border-t border-white/20">
                <div className="font-bold mb-1">🎯 All Objects Framed</div>
                <div className="text-xs text-green-300">
                  ✓ Every detected object has a bounding box
                </div>
                <div className="text-xs text-green-300">
                  ✓ Color-coded by distance (Red/Yellow/Green)
                </div>
                <div className="text-xs text-green-300">
                  ✓ Real-time movement tracking
                </div>
              </div>
            )}
            
            {/* Risk Assessment Summary */}
            {trackedObjects.length > 0 && (
              <div className="mt-3 pt-2 border-t border-white/20">
                <div className="font-bold mb-1">⚠️ Risk Levels</div>
                <div className="space-y-1 text-xs">
                  {trackedObjects.filter(obj => obj.riskLevel === 'critical').length > 0 && (
                    <div className="text-red-400">🔴 Critical: {trackedObjects.filter(obj => obj.riskLevel === 'critical').length}</div>
                  )}
                  {trackedObjects.filter(obj => obj.riskLevel === 'high').length > 0 && (
                    <div className="text-orange-400">🟠 High: {trackedObjects.filter(obj => obj.riskLevel === 'high').length}</div>
                  )}
                  {trackedObjects.filter(obj => obj.riskLevel === 'medium').length > 0 && (
                    <div className="text-yellow-400">🟡 Medium: {trackedObjects.filter(obj => obj.riskLevel === 'medium').length}</div>
                  )}
                  <div className="text-green-400">🟢 Safe: {trackedObjects.filter(obj => obj.riskLevel === 'low').length}</div>
                </div>
              </div>
            )}
            
            {/* Movement Analysis */}
            {trackedObjects.filter(obj => obj.velocity && obj.velocity > 0.5).length > 0 && (
              <div className="mt-3 pt-2 border-t border-white/20">
                <div className="font-bold mb-1">🚶 Movement</div>
                <div className="space-y-1 text-xs">
                  {trackedObjects.filter(obj => obj.movementDirection === 'approaching').length > 0 && (
                    <div className="text-red-400">→ Approaching: {trackedObjects.filter(obj => obj.movementDirection === 'approaching').length}</div>
                  )}
                  {trackedObjects.filter(obj => obj.movementDirection === 'receding').length > 0 && (
                    <div className="text-blue-400">← Receding: {trackedObjects.filter(obj => obj.movementDirection === 'receding').length}</div>
                  )}
                  {trackedObjects.filter(obj => obj.movementDirection === 'lateral').length > 0 && (
                    <div className="text-yellow-400">↔ Lateral: {trackedObjects.filter(obj => obj.movementDirection === 'lateral').length}</div>
                  )}
                </div>
              </div>
            )}
            
            {/* Proximity Warnings */}
            {trackedObjects.filter(obj => obj.proximityWarning).length > 0 && (
              <div className="mt-3 pt-2 border-t border-red-400 border-opacity-50">
                <div className="font-bold mb-1 text-red-400">🚨 Proximity Alerts</div>
                <div className="text-red-300 text-xs">
                  {trackedObjects.filter(obj => obj.proximityWarning).length} object(s) require immediate attention
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;