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
          console.log('🔍 Running detection on video...');
          
          const predictions = await modelRef.current.detect(video);
          console.log('🎯 Raw predictions:', predictions.length, predictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          
          // Higher threshold for more stable detection
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
        
        if (objectsToDraw.length > 0) {
          console.log('🎨 Drawing', objectsToDraw.length, 'tracked objects');
        }
        
        // Draw debug info to verify canvas is working
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 3;
        ctx.font = 'bold 20px Arial';
        ctx.fillStyle = '#FF0000';
        ctx.fillText(`Objects: ${objectsToDraw.length} | Total: ${trackedObjects.length}`, 10, 30);
        ctx.strokeRect(10, 40, 200, 30);
        
        // Draw additional debug info
        ctx.fillText(`Canvas: ${canvas.width}x${canvas.height}`, 10, 80);
        ctx.fillText(`Video: ${video.videoWidth}x${video.videoHeight}`, 10, 110);
        ctx.fillText(`Detection: ${currentDetections.length}`, 10, 140);
        
        // Draw a test pattern to verify canvas is working
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(10, 160, 100, 50);
        ctx.fillStyle = '#00FF00';
        ctx.fillText('TEST', 15, 185);
        
        // Draw grid lines to verify canvas positioning
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        for (let i = 0; i < canvas.width; i += 50) {
          ctx.beginPath();
          ctx.moveTo(i, 0);
          ctx.lineTo(i, canvas.height);
          ctx.stroke();
        }
        for (let i = 0; i < canvas.height; i += 50) {
          ctx.beginPath();
          ctx.moveTo(0, i);
          ctx.lineTo(canvas.width, i);
          ctx.stroke();
        }
        
        objectsToDraw.forEach((trackedObject, index) => {
          const [x, y, width, height] = trackedObject.bbox;
          console.log(`🎨 Drawing tracked object ${index}:`, trackedObject.class, 'at', [x, y, width, height]);
          
          // Different colors for focused vs stable objects
          if (trackedObject.focused) {
            ctx.strokeStyle = '#FFD700'; // Gold for focused objects
            ctx.fillStyle = '#FFD700';
            ctx.lineWidth = 6; // Thicker lines for focused objects
          } else {
            ctx.strokeStyle = '#00FF00'; // Green for stable objects
            ctx.fillStyle = '#00FF00';
            ctx.lineWidth = 4; // Normal thickness for stable objects
          }
          
          // Draw bounding box
          ctx.strokeRect(x, y, width, height);
          console.log(`✏️ Drew bounding box at [${x}, ${y}, ${width}, ${height}]`);
          
          // Draw comprehensive object information directly on canvas
          const distance = Math.round(trackedObject.distance || 0);
          const confidence = Math.round(trackedObject.confidence * 100);
          const direction = trackedObject.direction;
          const frameCount = trackedObject.frameCount;
          
          // Create detailed label with multiple lines
          const labelLines = [
            `${trackedObject.class.toUpperCase()}`,
            `${distance}m ${direction}`,
            `${confidence}% (${frameCount}f)`,
            trackedObject.focused ? 'FOCUSED' : 'Tracked'
          ];
          
          // Calculate label dimensions
          ctx.font = 'bold 16px Arial';
          const lineHeight = 20;
          const labelWidth = Math.max(...labelLines.map(line => ctx.measureText(line).width)) + 20;
          const labelHeight = labelLines.length * lineHeight + 10;
          
          // Position label above the object
          let labelX = x;
          let labelY = y - labelHeight - 10;
          
          // Adjust label position if it goes off-screen
          if (labelY < 0) {
            labelY = y + height + 10; // Put label below object
          }
          if (labelX + labelWidth > canvas.width) {
            labelX = canvas.width - labelWidth - 5; // Adjust for right edge
          }
          
          // Draw label background with semi-transparency
          ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
          ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
          
          // Draw label border
          ctx.strokeStyle = trackedObject.focused ? '#FFD700' : '#00FF00';
          ctx.lineWidth = 2;
          ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
          
          // Draw label text
          ctx.fillStyle = '#FFFFFF';
          labelLines.forEach((line, lineIndex) => {
            const lineY = labelY + 15 + (lineIndex * lineHeight);
            ctx.fillText(line, labelX + 10, lineY);
          });
          
          // Draw distance indicator line from object to label
          ctx.strokeStyle = trackedObject.focused ? '#FFD700' : '#00FF00';
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]); // Dashed line
          ctx.beginPath();
          ctx.moveTo(x + width/2, y);
          ctx.lineTo(labelX + labelWidth/2, labelY + labelHeight);
          ctx.stroke();
          ctx.setLineDash([]); // Reset to solid lines
          
          // Draw object center point
          ctx.fillStyle = trackedObject.focused ? '#FFD700' : '#00FF00';
          ctx.beginPath();
          ctx.arc(x + width/2, y + height/2, 4, 0, 2 * Math.PI);
          ctx.fill();
          
          // Draw confidence indicator bar
          const barWidth = 60;
          const barHeight = 6;
          const barX = x + width/2 - barWidth/2;
          const barY = y - 5;
          
          // Background bar
          ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.fillRect(barX, barY, barWidth, barHeight);
          
          // Confidence fill
          const confidenceWidth = (confidence / 100) * barWidth;
          if (confidence >= 80) {
            ctx.fillStyle = '#00FF00'; // Green for high confidence
          } else if (confidence >= 60) {
            ctx.fillStyle = '#FFFF00'; // Yellow for medium confidence
          } else {
            ctx.fillStyle = '#FF0000'; // Red for low confidence
          }
          ctx.fillRect(barX, barY, confidenceWidth, barHeight);
          
          // Bar border
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1;
          ctx.strokeRect(barX, barY, barWidth, barHeight);
          
          // Draw direction indicator arrow
          const arrowSize = 15;
          const arrowX = x + width/2;
          const arrowY = y + height + 15;
          
          ctx.strokeStyle = trackedObject.focused ? '#FFD700' : '#00FF00';
          ctx.lineWidth = 3;
          ctx.beginPath();
          
          if (direction === 'left') {
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX - arrowSize, arrowY);
            ctx.lineTo(arrowX - arrowSize + 5, arrowY - 5);
            ctx.moveTo(arrowX - arrowSize, arrowY);
            ctx.lineTo(arrowX - arrowSize + 5, arrowY + 5);
          } else if (direction === 'right') {
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX + arrowSize, arrowY);
            ctx.lineTo(arrowX + arrowSize - 5, arrowY - 5);
            ctx.moveTo(arrowX + arrowSize, arrowY);
            ctx.lineTo(arrowX + arrowSize - 5, arrowY + 5);
          } else { // center
            ctx.moveTo(arrowX - arrowSize/2, arrowY);
            ctx.lineTo(arrowX + arrowSize/2, arrowY);
            ctx.lineTo(arrowX, arrowY - 5);
          }
          ctx.stroke();
          
          // Add focus button for each object (only when not in focus mode)
          if (!focusMode) {
            const buttonWidth = 80;
            const buttonHeight = 25;
            const buttonX = x + width - buttonWidth - 5;
            const buttonY = y + height + 5;
            
            // Button background
            ctx.fillStyle = 'rgba(0, 123, 255, 0.9)';
            ctx.fillRect(buttonX, buttonY, buttonWidth, buttonHeight);
            
            // Button border
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1;
            ctx.strokeRect(buttonX, buttonY, buttonWidth, buttonHeight);
            
            // Button text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 12px Arial';
            ctx.fillText('FOCUS', buttonX + 20, buttonY + 17);
          }
          
          console.log(`✏️ Drew detailed label for ${trackedObject.class} at [${labelX}, ${labelY}]`);
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
        
        {/* Focus Mode Instructions */}
        {focusMode && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <Eye className="w-4 h-4 inline mr-2" />
              Focus Mode: Only focused objects are displayed and tracked. Click on objects to focus on them.
            </p>
          </div>
        )}
      </div>

      {/* Tracked Objects Panel */}
      {trackedObjects.length > 0 && (
        <div className="p-4 bg-card border-b border-border">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <Target className="w-5 h-5" />
            Tracked Objects ({trackedObjects.filter(obj => obj.stable).length} stable)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {trackedObjects
              .filter(obj => obj.stable && obj.confidence > 0.5)
              .sort((a, b) => b.confidence - a.confidence)
              .map((obj) => (
                <div
                  key={obj.id}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    obj.focused 
                      ? 'border-yellow-400 bg-yellow-50' 
                      : 'border-green-200 bg-green-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-lg capitalize">{obj.class}</span>
                    <Badge variant={obj.focused ? "default" : "secondary"}>
                      {Math.round(obj.confidence * 100)}%
                    </Badge>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>Direction: <span className="font-medium">{obj.direction}</span></div>
                    <div>Distance: <span className="font-medium">~{Math.round(obj.distance || 0)}m</span></div>
                    <div>Frames: <span className="font-medium">{obj.frameCount}</span></div>
                  </div>
                  {!obj.focused && (
                    <Button
                      size="sm"
                      onClick={() => focusOnObject(obj.id)}
                      className="w-full mt-2"
                      variant="outline"
                    >
                      <Focus className="w-3 h-3 mr-1" />
                      Focus
                    </Button>
                  )}
                  {obj.focused && (
                    <div className="mt-2 text-center">
                      <Badge variant="default" className="bg-yellow-500">
                        <Eye className="w-3 h-3 mr-1" />
                        Focused
                      </Badge>
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Enhanced Alert Panel */}
      <div className="p-4 bg-card border-b border-border">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Situational Awareness ({alerts.length} alerts)
        </h3>
        
        {/* Real-time Object Status */}
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-800 mb-2">Current Environment Status</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {trackedObjects.filter(obj => obj.stable && obj.confidence > 0.7).length}
              </div>
              <div className="text-blue-600">High Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {trackedObjects.filter(obj => obj.stable && obj.confidence > 0.5 && obj.confidence <= 0.7).length}
              </div>
              <div className="text-orange-600">Medium Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {trackedObjects.filter(obj => obj.distance && obj.distance < 5).length}
              </div>
              <div className="text-red-600">Very Close</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {trackedObjects.filter(obj => obj.distance && obj.distance >= 5 && obj.distance < 15).length}
              </div>
              <div className="text-green-600">Nearby</div>
            </div>
          </div>
        </div>

        {/* Active Alerts */}
        {alerts.length > 0 ? (
          <div className="space-y-2">
            {alerts
              .sort((a, b) => b.timestamp - a.timestamp)
              .slice(0, 5)
              .map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    alert.type === 'danger' 
                      ? 'border-red-500 bg-red-50 text-red-800' 
                      : alert.type === 'warning'
                      ? 'border-yellow-500 bg-yellow-50 text-yellow-800'
                      : 'border-blue-500 bg-blue-50 text-blue-800'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="font-medium">{alert.message}</span>
                  </div>
                  <div className="text-xs mt-1 opacity-75">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
          </div>
        ) : (
          <div className="text-center py-6 text-muted-foreground">
            <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No active alerts</p>
            <p className="text-sm">Objects will appear here when detected</p>
          </div>
        )}
      </div>

      {/* Comprehensive Object List for Situational Awareness */}
      <div className="p-4 bg-card border-b border-border">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Complete Environment Scan ({trackedObjects.length} objects detected)
        </h3>
        
        {trackedObjects.length > 0 ? (
          <div className="space-y-3">
            {/* High Priority Objects (Very Close) */}
            {trackedObjects.filter(obj => obj.distance && obj.distance < 3).length > 0 && (
              <div>
                <h4 className="font-medium text-red-700 mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  High Priority - Very Close Objects
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {trackedObjects
                    .filter(obj => obj.distance && obj.distance < 3)
                    .sort((a, b) => (a.distance || 0) - (b.distance || 0))
                    .map((obj) => (
                      <div key={obj.id} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-red-800 capitalize">{obj.class}</span>
                          <Badge variant="destructive" className="text-xs">
                            {Math.round(obj.distance || 0)}m
                          </Badge>
                        </div>
                        <div className="text-sm text-red-700 space-y-1">
                          <div>Direction: <span className="font-medium">{obj.direction}</span></div>
                          <div>Confidence: <span className="font-medium">{Math.round(obj.confidence * 100)}%</span></div>
                          <div>Stability: <span className="font-medium">{obj.frameCount} frames</span></div>
                          <div>Status: <span className="font-medium">{obj.focused ? 'FOCUSED' : 'Tracked'}</span></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Medium Priority Objects (Nearby) */}
            {trackedObjects.filter(obj => obj.distance && obj.distance >= 3 && obj.distance < 10).length > 0 && (
              <div>
                <h4 className="font-medium text-orange-700 mb-2 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Medium Priority - Nearby Objects
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {trackedObjects
                    .filter(obj => obj.distance && obj.distance >= 3 && obj.distance < 10)
                    .sort((a, b) => (a.distance || 0) - (b.distance || 0))
                    .map((obj) => (
                      <div key={obj.id} className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-orange-800 capitalize">{obj.class}</span>
                          <Badge variant="secondary" className="text-xs">
                            {Math.round(obj.distance || 0)}m
                          </Badge>
                        </div>
                        <div className="text-sm text-orange-700 space-y-1">
                          <div>Direction: <span className="font-medium">{obj.direction}</span></div>
                          <div>Confidence: <span className="font-medium">{Math.round(obj.confidence * 100)}%</span></div>
                          <div>Stability: <span className="font-medium">{obj.frameCount} frames</span></div>
                          <div>Status: <span className="font-medium">{obj.focused ? 'FOCUSED' : 'Tracked'}</span></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Low Priority Objects (Far Away) */}
            {trackedObjects.filter(obj => obj.distance && obj.distance >= 10).length > 0 && (
              <div>
                <h4 className="font-medium text-green-700 mb-2 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Low Priority - Distant Objects
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {trackedObjects
                    .filter(obj => obj.distance && obj.distance >= 10)
                    .sort((a, b) => (a.distance || 0) - (b.distance || 0))
                    .map((obj) => (
                      <div key={obj.id} className="p-3 bg-green-50 border border-green-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-green-800 capitalize">{obj.class}</span>
                          <Badge variant="outline" className="text-xs">
                            {Math.round(obj.distance || 0)}m
                          </Badge>
                        </div>
                        <div className="text-sm text-green-700 space-y-1">
                          <div>Direction: <span className="font-medium">{obj.direction}</span></div>
                          <div>Confidence: <span className="font-medium">{Math.round(obj.confidence * 100)}%</span></div>
                          <div>Stability: <span className="font-medium">{obj.frameCount} frames</span></div>
                          <div>Status: <span className="font-medium">{obj.focused ? 'FOCUSED' : 'Tracked'}</span></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Unstable Objects (Still being tracked) */}
            {trackedObjects.filter(obj => !obj.stable).length > 0 && (
              <div>
                <h4 className="font-medium text-gray-700 mb-2 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Unstable Objects (Still being tracked)
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {trackedObjects
                    .filter(obj => !obj.stable)
                    .map((obj) => (
                      <div key={obj.id} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-gray-800 capitalize">{obj.class}</span>
                          <Badge variant="outline" className="text-xs">
                            Unstable
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-700 space-y-1">
                          <div>Direction: <span className="font-medium">{obj.direction}</span></div>
                          <div>Confidence: <span className="font-medium">{Math.round(obj.confidence * 100)}%</span></div>
                          <div>Frames: <span className="font-medium">{obj.frameCount}/3</span></div>
                          <div>Distance: <span className="font-medium">~{Math.round(obj.distance || 0)}m</span></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-6 text-muted-foreground">
            <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No objects detected yet</p>
            <p className="text-sm">Start detection to begin scanning your environment</p>
          </div>
        )}
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
            border: '2px solid red',
            backgroundColor: 'rgba(255, 0, 0, 0.1)'
          }}
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
              toast({
                title: "Object Focused",
                description: `Now focusing on ${clickedObject.class}`,
              });
            }
          }}
        />
        
        {/* Enhanced Tracking Frame Overlay */}
        {isDetecting && (
          <div className="absolute top-4 left-4 bg-black bg-opacity-90 text-white p-4 rounded-lg border border-white/20 backdrop-blur-sm max-w-sm z-20">
            <div className="text-sm space-y-3">
              <div className="flex items-center gap-2 border-b border-white/20 pb-2">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                <span className="font-semibold">Live Detection Active</span>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-300">Model:</span>
                  <span className="font-medium">COCO-SSD</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">FPS:</span>
                  <span className="font-medium">{fps}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Objects:</span>
                  <span className="font-medium">{trackedObjects.filter(obj => obj.stable).length} stable</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Total:</span>
                  <span className="font-medium">{trackedObjects.length}</span>
                </div>
              </div>

              {/* Quick Object Summary */}
              {trackedObjects.filter(obj => obj.stable && obj.confidence > 0.6).length > 0 && (
                <div className="border-t border-white/20 pt-2">
                  <div className="text-xs text-gray-300 mb-1">Nearby Objects:</div>
                  <div className="space-y-1">
                    {trackedObjects
                      .filter(obj => obj.stable && obj.confidence > 0.6)
                      .sort((a, b) => (a.distance || 0) - (b.distance || 0))
                      .slice(0, 3)
                      .map((obj) => (
                        <div key={obj.id} className="flex justify-between items-center text-xs">
                          <span className="capitalize">{obj.class}</span>
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-1 rounded text-xs ${
                              obj.focused ? 'bg-yellow-600' : 'bg-green-600'
                            }`}>
                              {obj.focused ? 'FOCUS' : Math.round(obj.confidence * 100)}%
                            </span>
                            <span className="text-gray-300">
                              {Math.round(obj.distance || 0)}m {obj.direction}
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Focus Mode Status */}
              {focusMode && (
                <div className="border-t border-white/20 pt-2">
                  <div className="flex items-center gap-2 text-yellow-400">
                    <Eye className="w-4 h-4" />
                    <span className="text-xs font-medium">Focus Mode Active</span>
                  </div>
                  <div className="text-xs text-gray-300 mt-1">
                    Click objects to focus on them
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Focus Mode Instructions Overlay */}
        {focusMode && isDetecting && (
          <div className="absolute top-4 right-4 bg-black bg-opacity-75 text-white p-3 rounded-lg z-20">
            <p className="text-sm">
              <Eye className="w-4 h-4 inline mr-2" />
              Click on objects to focus on them
            </p>
          </div>
        )}

        {/* Canvas Status Indicator */}
        {isDetecting && (
          <div className="absolute bottom-4 left-4 bg-red-600 text-white p-2 rounded-lg z-20">
            <div className="text-xs font-bold">CANVAS ACTIVE</div>
            <div className="text-xs">Tracking Frame: {trackedObjects.length} objects</div>
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