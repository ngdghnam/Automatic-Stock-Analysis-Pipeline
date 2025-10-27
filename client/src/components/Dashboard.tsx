import React, { useState } from "react";
import { MultiSelect } from "./ui/multi-select";
import { Button } from "./ui/button";

const mockData = [{ value: "ABC", label: "ABC" }];

const Dashboard = () => {
  const [selectedValues, setSelectedValues] = useState<string[]>([]);
  return (
    <div>
      <div className="my-4 w-[24%] flex items-center">
        <MultiSelect
          options={mockData}
          onValueChange={setSelectedValues}
          defaultValue={selectedValues}
        ></MultiSelect>
        <Button className="mx-2 cursor-pointer" variant="default">
          Phân tích
        </Button>
      </div>
      <div>Analyse here</div>
    </div>
  );
};

export default Dashboard;
