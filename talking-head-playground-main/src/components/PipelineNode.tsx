import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface PipelineNodeProps {
  icon: LucideIcon;
  title: string;
  description: string;
  status: 'pending' | 'active' | 'completed';
  onClick?: () => void;
}

export const PipelineNode = ({ 
  icon: Icon, 
  title, 
  description, 
  status, 
  onClick 
}: PipelineNodeProps) => {
  return (
    <div 
      className={cn(
        "pipeline-node cursor-pointer",
        status === 'active' && "active",
        status === 'completed' && "completed"
      )}
      onClick={onClick}
    >
      <div className={cn(
        "w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300",
        status === 'pending' && "bg-muted/30",
        status === 'active' && "bg-primary text-primary-foreground animate-pulse-glow",
        status === 'completed' && "bg-secondary text-secondary-foreground"
      )}>
        <Icon size={24} />
      </div>
      <div className="text-center">
        <h3 className="font-semibold text-sm">{title}</h3>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      {status === 'active' && (
        <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
        </div>
      )}
    </div>
  );
};