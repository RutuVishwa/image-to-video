import { useState } from "react";
import { ArrowRight, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FirstTimeOverlayProps {
  onClose: () => void;
}

export const FirstTimeOverlay = ({ onClose }: FirstTimeOverlayProps) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      title: "Welcome to the Video Studio!",
      description: "Transform any image into a talking video with AI magic",
      position: "center"
    },
    {
      title: "Start Here",
      description: "Upload an image to begin your video creation journey",
      position: "center-top"
    },
    {
      title: "Follow the Pipeline",
      description: "Each step lights up as you progress through the creation process",
      position: "bottom-left"
    },
    {
      title: "Generate & Share",
      description: "Create multiple takes and download your favorite videos",
      position: "bottom"
    }
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="glass-card p-8 max-w-md mx-4 text-center relative">
        <Button
          variant="ghost"
          size="sm"
          className="absolute top-4 right-4"
          onClick={onClose}
        >
          <X size={16} />
        </Button>

        <div className="mb-6">
          <div className="w-16 h-16 mx-auto mb-4 glass-card rounded-full flex items-center justify-center animate-float">
            <span className="text-2xl">🎬</span>
          </div>
          <h2 className="text-xl font-bold mb-2 gradient-text">
            {steps[currentStep].title}
          </h2>
          <p className="text-muted-foreground">
            {steps[currentStep].description}
          </p>
        </div>

        <div className="flex justify-between items-center">
          <div className="flex gap-2">
            {steps.map((_, index) => (
              <div
                key={index}
                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                  index === currentStep ? 'bg-primary' : 'bg-muted/30'
                }`}
              />
            ))}
          </div>
          
          <Button onClick={handleNext} className="glass-button text-white">
            {currentStep < steps.length - 1 ? (
              <>
                Next <ArrowRight size={16} className="ml-2" />
              </>
            ) : (
              "Let's Start!"
            )}
          </Button>
        </div>
      </div>

      {/* Animated Dotted Path */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill="hsl(var(--primary))"
              opacity="0.6"
            />
          </marker>
        </defs>
        <path
          d="M 200 300 Q 400 200 600 300"
          stroke="hsl(var(--primary))"
          strokeWidth="2"
          strokeDasharray="5,5"
          fill="none"
          markerEnd="url(#arrowhead)"
          opacity="0.6"
          className="animate-pulse"
        />
      </svg>
    </div>
  );
};