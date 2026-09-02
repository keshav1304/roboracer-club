import { ButtonHTMLAttributes, ReactNode } from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'outline';
    size?: 'sm' | 'md' | 'lg';
    children: ReactNode;
}

const Button = ({
    className,
    variant = 'primary',
    size = 'md',
    children,
    ...props
}: ButtonProps) => {
    const baseStyles = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-brand-dark disabled:opacity-50 disabled:pointer-events-none';

    const variants = {
        primary: 'bg-brand-blue text-white hover:bg-brand-blue-hover',
        secondary: 'bg-brand-magenta text-white hover:bg-brand-magenta-hover',
        outline: 'border border-gray-600 text-gray-300 hover:bg-white/10 hover:text-white',
    };

    const sizes = {
        sm: 'h-9 px-3 text-xs',
        md: 'h-10 py-2 px-4 text-sm',
        lg: 'h-11 px-8 text-base',
    };

    return (
        <button
            className={cn(baseStyles, variants[variant], sizes[size], className)}
            {...props}
        >
            {children}
        </button>
    );
};

export default Button;
