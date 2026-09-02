"use client";

import { motion } from "framer-motion";

export default function ClubHeroTitle() {
    return (
        <div className="flex flex-col items-center justify-center mb-4">
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight text-center">
                <motion.span
                    initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="block text-gray-200 mb-2"
                >
                    Penn
                </motion.span>
                <motion.span
                    initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    transition={{ duration: 1.0, ease: "easeOut", delay: 0.3 }}
                    className="bg-clip-text text-transparent bg-gradient-to-r from-brand-blue to-brand-magenta"
                >
                    RoboRacer Club
                </motion.span>
            </h1>
        </div>
    );
}
