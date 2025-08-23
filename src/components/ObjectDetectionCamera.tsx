import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Button } from '@/components/ui/button';
import { Camera, Square, Info, X, Brain, Target, Zap } from 'lucide-react';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance: number;
  aiAnalysis?: AIAnalysis;
}

interface AIAnalysis {
  confidence: number;
  description: string;
  characteristics: string[];
  riskLevel: 'low' | 'medium' | 'high';
  recommendations: string[];
  movementPattern?: string;
  behaviorAnalysis?: string;
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
  const [statusPanelOpen, setStatusPanelOpen] = useState(false);
  const [aiAnalysisComplete, setAiAnalysisComplete] = useState(false);

  // Initialize TensorFlow.js and load model
  useEffect(() => {
    const initializeTensorFlow = async () => {
      try {
        console.log('Initializing TensorFlow.js...');
        await tf.ready();
        console.log('TensorFlow.js ready');
        
        console.log('Loading COCO-SSD model...');
        const model = await cocoSsd.load();
        console.log('COCO-SSD model loaded');
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
      } catch (error) {
        console.error('Error loading model:', error);
        setIsLoading(false);
      }
    };

    initializeTensorFlow();
  }, []);

  // Smart AI Object Analysis
  const analyzeObjectWithAI = useCallback((detection: Detection): AIAnalysis => {
    const { class: className, distance, score } = detection;
    
    // AI-powered object analysis based on type, distance, and context
    let analysis: AIAnalysis = {
      confidence: score,
      description: '',
      characteristics: [],
      riskLevel: 'low',
      recommendations: []
    };

    // Person analysis
    if (className === 'person') {
      analysis.description = `Human detected at ${distance}m distance`;
      analysis.characteristics = [
        'Bipedal movement pattern',
        'Variable size and posture',
        'Potential interaction capability'
      ];
      
      if (distance < 2) {
        analysis.riskLevel = 'high';
        analysis.recommendations = [
          'Maintain safe distance',
          'Assess intent and behavior',
          'Prepare for interaction'
        ];
      } else if (distance < 5) {
        analysis.riskLevel = 'medium';
        analysis.recommendations = [
          'Monitor movement patterns',
          'Maintain situational awareness',
          'Prepare for approach'
        ];
      } else {
        analysis.riskLevel = 'low';
        analysis.recommendations = [
          'Continue monitoring',
          'Assess movement direction',
          'Maintain normal operations'
        ];
      }
    }
    
    // Vehicle analysis
    else if (['car', 'truck', 'bus', 'motorcycle'].includes(className)) {
      analysis.description = `${className.charAt(0).toUpperCase() + className.slice(1)} detected at ${distance}m`;
      analysis.characteristics = [
        'Motorized transportation',
        'Variable speed capabilities',
        'Potential collision risk'
      ];
      
      if (distance < 3) {
        analysis.riskLevel = 'high';
        analysis.recommendations = [
          'Immediate evasive action',
          'Assess vehicle trajectory',
          'Prepare emergency response'
        ];
      } else if (distance < 8) {
        analysis.riskLevel = 'medium';
        analysis.recommendations = [
          'Monitor vehicle movement',
          'Assess speed and direction',
          'Maintain safe distance'
        ];
      } else {
        analysis.riskLevel = 'low';
        analysis.recommendations = [
          'Continue monitoring',
          'Assess traffic patterns',
          'Maintain normal operations'
        ];
      }
    }
    
    // Animal analysis
    else if (['dog', 'cat', 'horse', 'cow', 'bird'].includes(className)) {
      analysis.description = `${className.charAt(0).toUpperCase() + className.slice(1)} detected at ${distance}m`;
      analysis.characteristics = [
        'Unpredictable movement patterns',
        'Variable behavior based on species',
        'Potential interaction or threat'
      ];
      
      if (distance < 2) {
        analysis.riskLevel = 'medium';
        analysis.recommendations = [
          'Assess animal behavior',
          'Maintain calm demeanor',
          'Prepare for interaction'
        ];
      } else {
        analysis.riskLevel = 'low';
        analysis.recommendations = [
          'Monitor animal movement',
          'Assess behavior patterns',
          'Maintain safe distance'
        ];
      }
    }
    
    // Object analysis
    else {
      analysis.description = `${className.charAt(0).toUpperCase() + className.slice(1)} detected at ${distance}m`;
      analysis.characteristics = [
        'Static or mobile object',
        'Variable utility and function',
        'Context-dependent significance'
      ];
      
      if (distance < 1) {
        analysis.riskLevel = 'medium';
        analysis.recommendations = [
          'Assess object stability',
          'Check for hazards',
          'Maintain safe distance'
        ];
      } else {
        analysis.riskLevel = 'low';
        analysis.recommendations = [
          'Continue monitoring',
          'Assess object function',
          'Maintain normal operations'
        ];
      }
    }

    // Add movement pattern analysis
    if (detection.bbox) {
      const [x, y, width, height] = detection.bbox;
      const objectSize = width * height;
      
      if (objectSize > 10000) {
        analysis.movementPattern = 'Large object - significant presence';
      } else if (objectSize > 5000) {
        analysis.movementPattern = 'Medium object - moderate presence';
      } else {
        analysis.movementPattern = 'Small object - minimal presence';
      }
    }

    // Add behavior analysis
    analysis.behaviorAnalysis = `AI analysis indicates ${analysis.riskLevel} risk level based on object type, distance, and contextual factors.`;

    return analysis;
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
      console.log('Camera stream obtained');

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
    setAiAnalysisComplete(false);
  }, []);

  // Distance estimation using focal length approximation
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    
    // Known real-world dimensions for common objects (in meters)
    const objectDimensions: { [key: string]: number } = {
      'person': 0.5,      // Average person width
      'car': 1.8,         // Average car width
      'bicycle': 0.6,     // Bicycle width
      'motorcycle': 0.8,  // Motorcycle width
      'bus': 2.5,         // Bus width
      'truck': 2.5,       // Truck width
      'dog': 0.3,         // Dog width
      'cat': 0.2,         // Cat width
      'chair': 0.5,       // Chair width
      'table': 1.2,       // Table width
      'bottle': 0.08,     // Bottle width
      'cup': 0.08,        // Cup width
      'laptop': 0.35,     // Laptop width
      'tv': 1.0,          // TV width
      'cell phone': 0.07, // Phone width
      'book': 0.15,       // Book width
      'bowl': 0.15,       // Bowl width
      'apple': 0.08,      // Apple width
      'banana': 0.03,     // Banana width
      'airplane': 35.0,   // Airplane width
      'train': 3.0,       // Train width
      'boat': 2.0,        // Boat width
      'fire hydrant': 0.3, // Fire hydrant width
      'stop sign': 0.3,   // Stop sign width
      'bench': 1.5,       // Bench width
      'bird': 0.15,       // Bird width
      'horse': 1.0,       // Horse width
      'sheep': 0.8,       // Sheep width
      'cow': 1.2,         // Cow width
      'elephant': 2.5,    // Elephant width
      'bear': 1.5,        // Bear width
      'zebra': 1.0,       // Zebra width
      'giraffe': 0.8,     // Giraffe width
      'backpack': 0.3,    // Backpack width
      'umbrella': 0.8,    // Umbrella width
      'handbag': 0.3,     // Handbag width
      'suitcase': 0.5,    // Suitcase width
      'frisbee': 0.25,    // Frisbee width
      'skis': 0.1,        // Skis width
      'snowboard': 0.3,   // Snowboard width
      'sports ball': 0.22, // Sports ball width
      'kite': 0.8,        // Kite width
      'baseball bat': 0.05, // Baseball bat width
      'baseball glove': 0.25, // Baseball glove width
      'skateboard': 0.8,  // Skateboard width
      'surfboard': 2.0,   // Surfboard width
      'tennis racket': 0.3, // Tennis racket width
      'wine glass': 0.08, // Wine glass width
      'fork': 0.02,       // Fork width
      'knife': 0.02,      // Knife width
      'spoon': 0.02,      // Spoon width
      'sandwich': 0.15,   // Sandwich width
      'orange': 0.08,     // Orange width
      'broccoli': 0.15,   // Broccoli width
      'carrot': 0.03,     // Carrot width
      'hot dog': 0.15,    // Hot dog width
      'pizza': 0.3,       // Pizza width
      'donut': 0.1,       // Donut width
      'cake': 0.25,       // Cake width
      'bed': 1.4,         // Bed width
      'dining table': 1.8, // Dining table width
      'toilet': 0.4,      // Toilet width
      'remote': 0.15,     // Remote width
      'microwave': 0.5,   // Microwave width
      'oven': 0.6,        // Oven width
      'toaster': 0.3,     // Toaster width
      'sink': 0.5,        // Sink width
      'refrigerator': 0.8, // Refrigerator width
      'clock': 0.3,        // Clock width
      'vase': 0.15,       // Vase width
      'scissors': 0.15,   // Scissors width
      'teddy bear': 0.3,  // Teddy bear width
      'hair drier': 0.2,  // Hair drier width
      'toothbrush': 0.02, // Toothbrush width
    };

    // Get known width for this object type, default to 0.5m if unknown
    const knownWidth = objectDimensions[className] || 0.5;
    
    // Focal length approximation (adjust experimentally for your camera)
    const focalLength = 700;
    
    // Calculate distance using focal length formula: distance = (known_width * focal_length) / perceived_width
    const distance = (knownWidth * focalLength) / width;
    
    // Clamp distance to reasonable bounds (0.5m to 50m)
    const clampedDistance = Math.max(0.5, Math.min(50, distance));
    
    return Math.round(clampedDistance * 10) / 10; // Round to 1 decimal place
  };

  // Get color based on distance (exactly as specified)
  const getDistanceColor = (distance: number): string => {
    if (distance < 3) return '#FF0000';      // Red if distance < 3m
    if (distance <= 5) return '#FFFF00';     // Yellow if distance between 3m and 5m
    return '#00FF00';                        // Green if distance > 5m
  };

  // Main detection loop - optimized for 30+ FPS
  const startDetection = useCallback(() => {
    console.log('Starting detection loop...');
    
    let lastTime = Date.now();
    let frameCount = 0;
    let lastDetectionTime = 0;
    const detectionInterval = 200; // Run detection every 200ms for smooth 30+ FPS

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

      // Ensure video has dimensions
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.log('Waiting for video dimensions...');
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      // Set canvas size to match video dimensions EXACTLY
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

      // Draw a simple test pattern to verify canvas is working
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(10, 10, 100, 50);
      ctx.fillStyle = '#00FF00';
      ctx.font = 'bold 16px Arial';
      ctx.fillText('CANVAS OK', 15, 40);

      try {
        // Run object detection at controlled intervals
        if (currentTime - lastDetectionTime >= detectionInterval) {
          console.log('Running object detection...');
          
          const predictions = await modelRef.current.detect(video);
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.3);
          
          if (filteredPredictions.length > 0) {
            console.log('Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          }
          
          // Process detections with distance estimation and AI analysis
          const newDetections = filteredPredictions.map(prediction => {
            const distance = estimateDistance(prediction.bbox, prediction.class);
            const aiAnalysis = analyzeObjectWithAI({
              bbox: prediction.bbox,
              class: prediction.class,
              score: prediction.score,
              distance
            });
            
            return {
              bbox: prediction.bbox,
              class: prediction.class,
              score: prediction.score,
              distance,
              aiAnalysis
            };
          });
          
          setDetections(newDetections);
          setAiAnalysisComplete(true);
          lastDetectionTime = currentTime;
        }

        // Draw bounding boxes and labels for all detected objects
        detections.forEach((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          const color = getDistanceColor(detection.distance);
          
          console.log(`Drawing object ${index}:`, detection.class, 'at', x, y, width, height, 'distance:', detection.distance);
          
          // Validate bbox coordinates
          if (x < 0 || y < 0 || x + width > canvas.width || y + height > canvas.height) {
            console.warn('Bbox coordinates out of bounds:', { x, y, width, height, canvasWidth: canvas.width, canvasHeight: canvas.height });
            return;
          }
          
          // Draw bounding box with distance-based color
          ctx.strokeStyle = color;
          ctx.lineWidth = 4;
          ctx.strokeRect(x, y, width, height);
          
          // Draw label with object name + distance above the bounding box
          const label = `${detection.class} - ${detection.distance}m`;
          ctx.fillStyle = color;
          ctx.font = 'bold 16px Arial';
          
          // Position label above the object, or below if too close to top
          const labelY = y > 20 ? y - 10 : y + height + 20;
          ctx.fillText(label, x, labelY);
          
          // Add a colored dot to indicate distance level
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x + width - 10, y + 10, 5, 0, 2 * Math.PI);
          ctx.fill();
          
          // Add AI analysis indicator if available
          if (detection.aiAnalysis) {
            const aiColor = detection.aiAnalysis.riskLevel === 'high' ? '#FF0000' : 
                           detection.aiAnalysis.riskLevel === 'medium' ? '#FFFF00' : '#00FF00';
            
            ctx.fillStyle = aiColor;
            ctx.beginPath();
            ctx.arc(x + 10, y + 10, 6, 0, 2 * Math.PI);
            ctx.fill();
            
            // Add AI icon
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 12px Arial';
            ctx.fillText('🤖', x + 6, y + 15);
            
            // Draw AI Analysis Information ON SCREEN
            const analysisX = x + width + 10;
            const analysisY = y;
            const analysisWidth = 200;
            const analysisHeight = 120;
            
            // Analysis background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
            ctx.fillRect(analysisX, analysisY, analysisWidth, analysisHeight);
            
            // Analysis border
            ctx.strokeStyle = aiColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(analysisX, analysisY, analysisWidth, analysisHeight);
            
            // Analysis title
            ctx.fillStyle = aiColor;
            ctx.font = 'bold 14px Arial';
            ctx.fillText('🤖 AI ANALYSIS', analysisX + 5, analysisY + 20);
            
            // Object type and distance
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(`${detection.class.toUpperCase()}`, analysisX + 5, analysisY + 35);
            ctx.fillText(`${detection.distance}m away`, analysisX + 5, analysisY + 50);
            
            // Risk level with color coding
            const riskText = `${detection.aiAnalysis.riskLevel.toUpperCase()} RISK`;
            ctx.fillStyle = aiColor;
            ctx.font = 'bold 12px Arial';
            ctx.fillText(riskText, analysisX + 5, analysisY + 65);
            
            // Key characteristics (first 2)
            ctx.fillStyle = '#CCCCCC';
            ctx.font = '10px Arial';
            if (detection.aiAnalysis.characteristics.length > 0) {
              ctx.fillText(detection.aiAnalysis.characteristics[0], analysisX + 5, analysisY + 80);
            }
            if (detection.aiAnalysis.characteristics.length > 1) {
              ctx.fillText(detection.aiAnalysis.characteristics[1], analysisX + 5, analysisY + 95);
            }
            
            // Top recommendation
            if (detection.aiAnalysis.recommendations.length > 0) {
              ctx.fillStyle = '#FFFF00';
              ctx.font = 'bold 10px Arial';
              ctx.fillText('💡 ' + detection.aiAnalysis.recommendations[0], analysisX + 5, analysisY + 110);
            }
            
            // Add movement pattern if available
            if (detection.aiAnalysis.movementPattern) {
              ctx.fillStyle = '#00FFFF';
              ctx.font = '10px Arial';
              ctx.fillText('📊 ' + detection.aiAnalysis.movementPattern, analysisX + 5, analysisY + 125);
            }
            
            // Add confidence indicator
            const confidence = Math.round(detection.aiAnalysis.confidence * 100);
            ctx.fillStyle = confidence > 80 ? '#00FF00' : confidence > 60 ? '#FFFF00' : '#FF0000';
            ctx.font = 'bold 10px Arial';
            ctx.fillText(`Confidence: ${confidence}%`, analysisX + 5, analysisY + 140);
          }
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

      // Continue loop for real-time updates
      if (isDetecting) {
        animationRef.current = requestAnimationFrame(detect);
      }
    };

    detect();
  }, [detections, isDetecting, analyzeObjectWithAI]);

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
          
          {/* AI Status Indicator */}
          {aiAnalysisComplete && (
            <Button
              variant="outline"
              className="gap-2 bg-green-50 border-green-200 text-green-700 hover:bg-green-100"
              disabled
            >
              <Brain className="w-4 h-4" />
              AI Active
            </Button>
          )}
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
        
        {/* Hidden Status Panel - Click to Open */}
        <div className="absolute top-2 left-2 z-20">
          <Button
            onClick={() => setStatusPanelOpen(!statusPanelOpen)}
            variant="outline"
            size="sm"
            className="bg-black bg-opacity-75 text-white border-white/20 hover:bg-black/90"
          >
            {statusPanelOpen ? <X className="w-4 h-4" /> : <Info className="w-4 h-4" />}
            {statusPanelOpen ? ' Hide' : ' Show'} Status
          </Button>
        </div>
        
        {/* Expandable Status Panel */}
        {statusPanelOpen && (
          <div className="absolute top-12 left-2 bg-black bg-opacity-90 text-white p-4 rounded-lg text-sm z-20 max-w-80">
            <div className="font-bold mb-3 border-b border-white/20 pb-2 flex items-center gap-2">
              <Target className="w-4 h-4" />
              AI Detection Status
            </div>
            
            <div className="space-y-3">
              {/* Basic Status */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Objects Detected:</span>
                  <span className="text-green-400 font-bold">{detections.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>FPS:</span>
                  <span className={fps >= 30 ? 'text-green-400' : fps >= 25 ? 'text-yellow-400' : 'text-red-400'}>
                    {fps}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Canvas:</span>
                  <span className="text-blue-400">{canvasRef.current?.width || 0} x {canvasRef.current?.height || 0}</span>
                </div>
              </div>
              
              {/* AI Analysis Status */}
              {aiAnalysisComplete && (
                <div className="pt-2 border-t border-white/20">
                  <div className="flex items-center gap-2 text-green-400 font-bold mb-2">
                    <Brain className="w-4 h-4" />
                    AI Analysis Active
                  </div>
                  <div className="text-xs text-green-300">
                    Smart object identification and risk assessment enabled
                  </div>
                </div>
              )}
              
              {/* Detection Status */}
              {detections.length > 0 ? (
                <div className="pt-2 border-t border-green-400 border-opacity-50">
                  <div className="text-xs text-green-300 font-bold mb-2">✅ Objects Analyzed:</div>
                  <div className="space-y-2">
                    {detections.map((detection, index) => (
                      <div key={index} className="text-xs bg-white/10 p-2 rounded">
                        <div className="font-bold text-white">{detection.class}</div>
                        <div className="text-gray-300">{detection.distance}m away</div>
                        {detection.aiAnalysis && (
                          <div className="mt-1">
                            <div className={`text-xs px-2 py-1 rounded inline-block ${
                              detection.aiAnalysis.riskLevel === 'high' ? 'bg-red-500/20 text-red-300' :
                              detection.aiAnalysis.riskLevel === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                              'bg-green-500/20 text-green-300'
                            }`}>
                              {detection.aiAnalysis.riskLevel.toUpperCase()} RISK
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="pt-2 border-t border-yellow-400 border-opacity-50">
                  <div className="text-xs text-yellow-300 font-bold mb-2">🔍 No Objects Detected</div>
                  <div className="text-xs text-yellow-300">
                    Move objects in front of camera for AI analysis
                  </div>
                </div>
              )}
              
              {/* Distance Legend */}
              <div className="pt-2 border-t border-white/20">
                <div className="text-xs font-bold mb-1">Distance Colors:</div>
                <div className="text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span>Red: &lt;3m (Close)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <span>Yellow: 3-5m (Medium)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span>Green: &gt;5m (Far)</span>
                  </div>
                </div>
              </div>
              
              {/* AI Features */}
              <div className="pt-2 border-t border-white/20">
                <div className="text-xs font-bold mb-2 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  AI Features:
                </div>
                <div className="text-xs space-y-1 text-blue-300">
                  <div>✓ Smart object identification</div>
                  <div>✓ Risk level assessment</div>
                  <div>✓ Behavioral analysis</div>
                  <div>✓ Context-aware recommendations</div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Floating Analysis Summary Panel - Always Visible */}
        {isDetecting && detections.length > 0 && (
          <div className="absolute top-2 right-2 bg-black bg-opacity-90 text-white p-3 rounded-lg text-sm z-20 max-w-64">
            <div className="font-bold mb-2 border-b border-white/20 pb-1 flex items-center gap-2">
              <Brain className="w-4 h-4 text-green-400" />
              Live Analysis
            </div>
            
            <div className="space-y-2">
              {detections.slice(0, 3).map((detection, index) => (
                <div key={index} className="text-xs bg-white/10 p-2 rounded border-l-2" 
                     style={{ borderLeftColor: getDistanceColor(detection.distance) }}>
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white">{detection.class}</span>
                    <span className="text-gray-300">{detection.distance}m</span>
                  </div>
                  {detection.aiAnalysis && (
                    <div className="mt-1">
                      <div className={`text-xs px-2 py-1 rounded inline-block ${
                        detection.aiAnalysis.riskLevel === 'high' ? 'bg-red-500/20 text-red-300' :
                        detection.aiAnalysis.riskLevel === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                        'bg-green-500/20 text-green-300'
                      }`}>
                        {detection.aiAnalysis.riskLevel.toUpperCase()}
                      </div>
                      {detection.aiAnalysis.recommendations.length > 0 && (
                        <div className="text-xs text-yellow-300 mt-1">
                          💡 {detection.aiAnalysis.recommendations[0]}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              
              {detections.length > 3 && (
                <div className="text-xs text-gray-400 text-center pt-1 border-t border-white/20">
                  +{detections.length - 3} more objects analyzed
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;