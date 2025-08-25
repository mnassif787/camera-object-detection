import { useState, useRef, useCallback } from 'react';

interface VoiceAlertOptions {
  enabled: boolean;
  volume: number;
  rate: number;
  pitch: number;
  alertType: 'immediate' | 'summary' | 'both';
  announceDistance: boolean;
  announceDirection: boolean;
}

interface Detection {
  class: string;
  distance: number;
  direction: string;
  score: number;
}

const useVoiceAlerts = () => {
  const [isSupported, setIsSupported] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [options, setOptions] = useState<VoiceAlertOptions>({
    enabled: true,
    volume: 0.8,
    rate: 1.0,
    pitch: 1.0,
    alertType: 'immediate',
    announceDistance: true,
    announceDirection: true,
  });

  const speechRef = useRef<SpeechSynthesis | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Initialize speech synthesis
  const initializeSpeech = useCallback(() => {
    if ('speechSynthesis' in window) {
      speechRef.current = window.speechSynthesis;
      setIsSupported(true);
      
      // Get available voices
      const loadVoices = () => {
        const voices = speechRef.current?.getVoices() || [];
        // Prefer English voices
        const englishVoice = voices.find(voice => 
          voice.lang.startsWith('en') && voice.default
        ) || voices.find(voice => 
          voice.lang.startsWith('en')
        ) || voices[0];
        
        if (englishVoice) {
          console.log('Using voice:', englishVoice.name);
        }
      };

      // Load voices when they become available
      if (speechRef.current.onvoiceschanged !== undefined) {
        speechRef.current.onvoiceschanged = loadVoices;
      }
      loadVoices();
    } else {
      setIsSupported(false);
      console.warn('Speech synthesis not supported in this browser');
    }
  }, []);

  // Speak text with current options
  const speak = useCallback((text: string, priority: 'high' | 'normal' = 'normal') => {
    if (!isSupported || !options.enabled || !speechRef.current) return;

    // Cancel any ongoing speech if this is a high priority alert
    if (priority === 'high' && isSpeaking) {
      speechRef.current.cancel();
    }

    // Create new utterance
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.volume = options.volume;
    utterance.rate = options.rate;
    utterance.pitch = options.pitch;
    
    // Set voice if available
    const voices = speechRef.current.getVoices();
    const englishVoice = voices.find(voice => 
      voice.lang.startsWith('en') && voice.default
    ) || voices.find(voice => 
      voice.lang.startsWith('en')
    ) || voices[0];
    
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    // Event handlers
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = (event) => {
      console.error('Speech synthesis error:', event);
      setIsSpeaking(false);
    };

    // Store reference and speak
    utteranceRef.current = utterance;
    speechRef.current.speak(utterance);
  }, [isSupported, options, isSpeaking]);

  // Announce immediate detection
  const announceDetection = useCallback((detection: Detection) => {
    if (options.alertType === 'summary') return;

    let message = `Detected ${detection.class}`;
    
    if (options.announceDistance) {
      message += ` at ${detection.distance.toFixed(1)} meters`;
    }
    
    if (options.announceDirection) {
      message += ` to the ${detection.direction.toLowerCase()}`;
    }

    speak(message, 'high');
  }, [options, speak]);

  // Announce summary of detections
  const announceSummary = useCallback((detections: Detection[]) => {
    if (options.alertType === 'immediate') return;

    if (detections.length === 0) {
      speak('No objects detected');
      return;
    }

    const objectCounts: { [key: string]: number } = {};
    detections.forEach(detection => {
      objectCounts[detection.class] = (objectCounts[detection.class] || 0) + 1;
    });

    const summary = Object.entries(objectCounts)
      .map(([className, count]) => `${count} ${className}${count > 1 ? 's' : ''}`)
      .join(', ');

    speak(`Detected ${summary}`);
  }, [options, speak]);

  // Stop all speech
  const stopSpeaking = useCallback(() => {
    if (speechRef.current) {
      speechRef.current.cancel();
      setIsSpeaking(false);
    }
  }, []);

  // Update options
  const updateOptions = useCallback((newOptions: Partial<VoiceAlertOptions>) => {
    setOptions(prev => ({ ...prev, ...newOptions }));
  }, []);

  // Test voice
  const testVoice = useCallback(() => {
    speak('Voice alerts are working correctly. You will hear announcements when objects are detected.');
  }, [speak]);

  return {
    isSupported,
    isSpeaking,
    options,
    initializeSpeech,
    announceDetection,
    announceSummary,
    stopSpeaking,
    updateOptions,
    testVoice,
  };
};

export default useVoiceAlerts;
export type { VoiceAlertOptions, Detection };