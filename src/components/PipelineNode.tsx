import { LucideIcon } from "lucide-react";

interface PipelineNodeProps {
  icon: LucideIcon;
  title: string;
  description: string;
  status: 'completed' | 'active' | 'pending';
  onClick: () => void;
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
      className={`pipeline-node cursor-pointer ${status}`}
      onClick={onClick}
    >
      <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 ${
        status === 'completed' 
          ? 'bg-primary text-white' 
          : status === 'active' 
          ? 'bg-primary/20 text-primary ring-2 ring-primary/50 dark:bg-primary/30 dark:ring-primary/40' 
          : 'bg-muted/50 text-muted-foreground dark:bg-gray-600/50'
      }`}>
        <Icon size={24} />
      </div>
      <div className="text-center">
        <h4 className={`text-sm font-medium ${
          status === 'completed' || status === 'active' 
            ? 'text-foreground' 
            : 'text-muted-foreground'
        }`}>
          {title}
        </h4>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
    </div>
  );
};

