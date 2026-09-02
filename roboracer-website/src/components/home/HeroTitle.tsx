"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

export default function HeroTitle() {
    const [text, setText] = useState("");
    const fullText = "-mini";

    useEffect(() => {
        const startDelay = 1500; // Wait for "RoboRacer" to fade in
        const typingSpeed = 150;
        let timeoutId: NodeJS.Timeout;
        let intervalId: NodeJS.Timeout;

        timeoutId = setTimeout(() => {
            let currentIndex = 0;
            intervalId = setInterval(() => {
                if (currentIndex < fullText.length) {
                    currentIndex++;
                    setText(fullText.slice(0, currentIndex));
                } else {
                    clearInterval(intervalId);
                }
            }, typingSpeed);
        }, startDelay);

        return () => {
            clearTimeout(timeoutId);
            if (intervalId) clearInterval(intervalId);
        };
    }, []);

    return (
        <div className="flex flex-col items-center justify-center mb-6">
            <h1 className="text-6xl md:text-8xl font-bold tracking-tight flex items-baseline">
                <motion.span
                    initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    transition={{ duration: 1.0, ease: "easeOut" }}
                    className="bg-clip-text text-transparent bg-gradient-to-r from-brand-blue to-brand-magenta"
                >
                    RoboRacer
                </motion.span>

                <span className="bg-clip-text text-transparent bg-gradient-to-r from-brand-magenta to-white ml-1">
                    {text}
                    <motion.span
                        animate={{ opacity: [0, 1, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8 }}
                        className="inline-block w-[3px] h-[0.8em] bg-brand-magenta ml-1 align-middle"
                        style={{ verticalAlign: "middle" }}
                    />
                </span>
            </h1>
        </div>
    );
}
