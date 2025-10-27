import React from "react";
import { Button } from "./ui/button";
import { ArrowDownToLine, ChartNoAxesCombined } from "lucide-react";
import { ModeToggle } from "./mode-toggle";

const Navbar = () => {
  return (
    <div className="w-full flex justify-between items-center">
      <div className="flex flex-col gap-1">
        <h1 className="font-semibold text-3xl">
          Welcome to R/Python Dashboard
        </h1>
        <p>
          <b>Instructor:</b> Trieu Viet Cuong - <b>Class code:</b> 251IS2901
        </p>
      </div>
      <div className="flex items-center">
        <Button className="mr-2 cursor-pointer" variant="outline">
          Export Data
          <ArrowDownToLine />
        </Button>
        <Button className="cursor-pointer mr-2">
          Create Report
          <ChartNoAxesCombined />
        </Button>
        <ModeToggle />
      </div>
    </div>
  );
};

export default Navbar;
