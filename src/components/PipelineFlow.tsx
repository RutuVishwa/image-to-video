import { Image as ImageIcon, Mic, Type, Sparkles } from "lucide-react";
import { PipelineNode } from "./PipelineNode";

interface PipelineFlowProps {
  currentStep: number;
  onStepClick: (step: number) => void;
}

export const PipelineFlow = ({ currentStep, onStepClick }: PipelineFlowProps) => {
  const steps = [
    {
      icon: ImageIcon,
      title: "Upload Image",
      description: "Choose your photo",
      status: (currentStep > 0 ? 'completed' : currentStep === 0 ? 'active' : 'pending') as 'completed' | 'active' | 'pending'
    },
    {
      icon: Mic,
      title: "Add Voice",
      description: "Upload audio file",
      status: (currentStep > 1 ? 'completed' : currentStep === 1 ? 'active' : 'pending') as 'completed' | 'active' | 'pending'
    },
    {
      icon: Type,
      title: "Enter Text",
      description: "Script to speak",
      status: (currentStep > 2 ? 'completed' : currentStep === 2 ? 'active' : 'pending') as 'completed' | 'active' | 'pending'
    },
    {
      icon: Sparkles,
      title: "Generate",
      description: "Create video",
      status: (currentStep > 3 ? 'completed' : currentStep === 3 ? 'active' : 'pending') as 'completed' | 'active' | 'pending'
    }
  ];

  return (
    <div className="glass-card p-6">
      <h3 className="font-semibold mb-6 text-center gradient-text">Creation Pipeline</h3>
      
      <div className="flex items-center justify-between relative">
        {steps.map((step, index) => (
          <div key={index} className="flex flex-col items-center relative z-10">
            <PipelineNode
              icon={step.icon}
              title={step.title}
              description={step.description}
              status={step.status}
              onClick={() => onStepClick(index)}
            />
            
            {index < steps.length - 1 && (
              <div className="absolute top-6 left-full w-full flex items-center justify-center">
                <div 
                  className={`pipeline-connection w-20 ${
                    currentStep > index ? 'active' : ''
                  }`}
                ></div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Progress Indicator */}
      <div className="mt-8 space-y-2">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Progress</span>
          <span>{Math.round((currentStep / (steps.length - 1)) * 100)}%</span>
        </div>
        <div className="w-full bg-muted/30 rounded-full h-2 overflow-hidden dark:bg-gray-700/30">
          <div 
            className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-700 ease-out"
            style={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

